"""Depth-correct projection: draw the camera-space image from the projector's viewpoint over the depth map.

Replaces `Calibration.to_projector` on the 3D path. Every camera pixel (u, v) with depth d is the 3D point
X = d K_cam^-1 [u, v, 1] (OpenCV camera coords, +z forward). It lands in the projector at
p = K_p distort((R X + t) / z) with Brown-Conrady `dist_p` = (k1, k2, p1, p2, k3) applied to the normalised
coordinates, exactly as `cv2.projectPoints`. The projector frame shows, at p, the colour of `image_cam` at (u, v).

GPU design (ModernGL, offscreen, one standalone OpenGL 4.1 context per instance):
- Mesh pass: a triangle grid over the depth map, one vertex per `mesh_step` depth pixel, two triangles per grid
  quad. There are no vertex or index buffers: a non-indexed draw of 6 vertices per quad, and the vertex shader
  derives its triangle and corner from `gl_VertexID`, fetches that triangle's three depths from an R32F texture,
  and projects all three corners (distortion in the shader). A triangle is dropped when any corner has no valid
  depth (0 / NaN / `valid` false), lies behind the projector, or is outside the radius where the distortion
  polynomial is trusted: all three vertices move outside the clip volume, so the GPU discards it. Nothing is
  rebuilt per frame, the CPU only uploads two textures. A triangle whose three depths span more than
  `depth_gap_m` is a "curtain" across a real depth discontinuity: it is drawn black, flagged as a gap, and pushed
  to the far end of the depth range so it can never hide a real surface. Real surfaces sample `image_cam` with the
  vertex's own (u, v) as texture coordinates (bilinear). The depth test keeps the nearest surface (an arm in front
  of the torso wins).
- Hole fill (two full-screen passes): a morphological close of the coverage mask (surfaces + curtains), radius
  `fill_holes_px // 2`, so empty gaps at most `fill_holes_px` wide are filled with the mean colour of the surface
  pixels around them. Curtain pixels count as covered and are never filled, so the fill can never bridge a real
  depth gap however narrow it is in projector pixels.
- Readback: the final pass writes an RGB8 texture that is read straight into the reused output buffer. The image
  texture holds BGR bytes and the shaders never swap channels, so the readback is already BGR. Framebuffer row y
  is projector row y (no flip): `glReadPixels` returns row 0 first, which is OpenCV's top row.

A context is current only on the thread that uses it: construct and call `render()` from the same thread.
"""
from __future__ import annotations

import logging
import time

import moderngl
import numpy as np

logger = logging.getLogger(__name__)

# Keys not (yet) in config.yaml `render3d:`; used only when the config omits them.
_DEFAULTS = {
    # Triangles beyond this multiple of the projector image's corner radius (normalised, undistorted) are
    # dropped: they cannot reach the image, and far off-axis the distortion polynomial folds back onto it.
    "distortion_radius_factor": 1.5,
}

_N_DIST = 5            # Brown-Conrady k1, k2, p1, p2, k3 (OpenCV order)
_RADIUS_SAMPLES = 2048  # samples when searching the radius where the radial distortion stops being monotone

_MESH_VS = """
#version 410
uniform sampler2D depth_tex;   // R32F metres, camera resolution, 0 = invalid
uniform int step;              // mesh_step: depth pixels between grid vertices
uniform int quads_x;           // grid quads per row
uniform vec2 cam_size;         // (W, H)
uniform vec2 proj_size;        // (pw, ph)
uniform mat3 Kc_inv;
uniform mat3 R;
uniform vec3 t;
uniform mat3 Kp;
uniform vec4 k12p12;           // k1, k2, p1, p2
uniform float k3;
uniform float r2_max;          // trusted squared radius of the normalised projector coordinates
uniform float gap;             // depth_gap_m

out vec2 tc;
flat out float surface;        // 1 = real surface, 0 = curtain across a depth gap
invariant gl_Position;

const ivec2 CORNER[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 0), ivec2(1, 1), ivec2(0, 1));
const float CURTAIN_NDC_Z = 0.99999;   // behind every real surface (those stay below 1 - 2 / (z + 1))

// Clip-space projector position of camera pixel `uv` at depth `d`; w < 0 means "cannot be projected".
vec4 project(vec2 uv, float d) {
    vec3 x = R * (d * (Kc_inv * vec3(uv, 1.0))) + t;
    if (!(x.z > 0.0)) return vec4(0.0, 0.0, 0.0, -1.0);
    vec2 n = x.xy / x.z;
    float r2 = dot(n, n);
    if (!(r2 <= r2_max)) return vec4(0.0, 0.0, 0.0, -1.0);
    float radial = 1.0 + r2 * (k12p12.x + r2 * (k12p12.y + r2 * k3));
    vec2 nd = n * radial + vec2(2.0 * k12p12.z * n.x * n.y + k12p12.w * (r2 + 2.0 * n.x * n.x),
                                k12p12.z * (r2 + 2.0 * n.y * n.y) + 2.0 * k12p12.w * n.x * n.y);
    vec3 p = Kp * vec3(nd, 1.0);
    vec2 ndc = (p.xy / p.z + 0.5) / proj_size * 2.0 - 1.0;   // pixel centre i -> NDC; row 0 = framebuffer row 0
    float zndc = 1.0 - 2.0 / (x.z + 1.0);                    // monotone in depth, inside (-1, 1) for z > 0
    return vec4(ndc * x.z, zndc * x.z, x.z);                 // w = depth: perspective-correct texcoords
}

void main() {
    int quad = gl_VertexID / 6;
    int k = gl_VertexID - quad * 6;
    int first = k < 3 ? 0 : 3;
    ivec2 base = ivec2(quad % quads_x, quad / quads_x) * step;
    vec4 pos[3];
    float dmin = 1e30;
    float dmax = -1e30;
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        ivec2 c = base + CORNER[first + i] * step;
        float d = texelFetch(depth_tex, c, 0).r;
        ok = ok && d > 0.0;                     // also false for NaN
        pos[i] = project(vec2(c), d);
        ok = ok && pos[i].w > 0.0;
        dmin = min(dmin, d);
        dmax = max(dmax, d);
    }
    if (!ok) {                                  // whole triangle outside the clip volume: discarded
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        tc = vec2(0.0);
        surface = 0.0;
        return;
    }
    vec4 me = pos[k - first];
    tc = (vec2(base + CORNER[k] * step) + 0.5) / cam_size;
    if (dmax - dmin > gap) {
        surface = 0.0;
        gl_Position = vec4(me.xy, CURTAIN_NDC_Z * me.w, me.w);
    } else {
        surface = 1.0;
        gl_Position = me;
    }
}
"""

_MESH_FS = """
#version 410
uniform sampler2D image_tex;   // BGR bytes, camera resolution
in vec2 tc;
flat in float surface;
out vec4 color;                // rgb = BGR colour, a = 1 surface / 0.5 curtain / 0 empty
void main() {
    color = surface > 0.5 ? vec4(texture(image_tex, tc).rgb, 1.0) : vec4(0.0, 0.0, 0.0, 0.5);
}
"""

_FULLSCREEN_VS = """
#version 410
void main() {
    gl_Position = vec4(float((gl_VertexID & 1) * 4 - 1), float((gl_VertexID >> 1) * 4 - 1), 0.0, 1.0);
}
"""

_DILATE_FS = """
#version 410
uniform sampler2D mesh_tex;
uniform int radius;
out vec4 color;                // rgb = mean surface colour in the window, a = 1 if anything covers the window
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(mesh_tex, 0);
    vec3 sum = vec3(0.0);
    float n = 0.0;
    float cover = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            ivec2 q = p + ivec2(dx, dy);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) continue;
            vec4 s = texelFetch(mesh_tex, q, 0);
            if (s.a > 0.25) cover = 1.0;
            if (s.a > 0.75) { sum += s.rgb; n += 1.0; }
        }
    }
    color = vec4(n > 0.0 ? sum / n : vec3(0.0), cover);
}
"""

_RESOLVE_FS = """
#version 410
uniform sampler2D mesh_tex;
uniform sampler2D dil_tex;
uniform int radius;
out vec4 color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 s = texelFetch(mesh_tex, p, 0);
    color = vec4(0.0, 0.0, 0.0, 1.0);
    if (s.a > 0.75) { color = vec4(s.rgb, 1.0); return; }
    if (s.a > 0.25 || radius == 0) return;             // curtain (real gap) or fill disabled: stays dark
    ivec2 size = textureSize(dil_tex, 0);
    for (int dy = -radius; dy <= radius; ++dy) {        // erosion of the dilated coverage = closing
        for (int dx = -radius; dx <= radius; ++dx) {
            ivec2 q = p + ivec2(dx, dy);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) return;
            if (texelFetch(dil_tex, q, 0).a < 0.5) return;
        }
    }
    color = vec4(texelFetch(dil_tex, p, 0).rgb, 1.0);
}
"""

# Texture units (fixed; this class owns its context).
_UNIT_DEPTH, _UNIT_IMAGE, _UNIT_MESH, _UNIT_DIL = 0, 1, 2, 3


def _size(s) -> tuple[int, int]:
    w, h = (int(v) for v in s)
    if w <= 1 or h <= 1:
        raise ValueError(f"size must be (w, h) with both > 1, got {s}")
    return w, h


def _mat3(M, name: str) -> np.ndarray:
    M = np.asarray(M, np.float64)
    if M.shape != (3, 3) or not np.all(np.isfinite(M)):
        raise ValueError(f"{name} must be a finite 3x3 matrix, got shape {M.shape}")
    return M


def _dist5(dist_p) -> np.ndarray:
    """(k1, k2, p1, p2, k3) from an OpenCV distortion vector; longer models (rational, thin prism) are refused."""
    d = np.zeros(_N_DIST, np.float64)
    if dist_p is None:
        return d
    v = np.asarray(dist_p, np.float64).ravel()
    if not np.all(np.isfinite(v)):
        raise ValueError(f"dist_p must be finite, got {v}")
    if v.size > _N_DIST and np.any(v[_N_DIST:] != 0):
        raise ValueError(f"dist_p has {v.size} coefficients with non-zero terms beyond k3; only the 5-term "
                         "Brown-Conrady model (k1, k2, p1, p2, k3) is implemented")
    d[:min(v.size, _N_DIST)] = v[:_N_DIST]
    return d


def _trusted_r2(K_p: np.ndarray, dist: np.ndarray, proj_size: tuple[int, int], factor: float) -> float:
    """Squared normalised radius inside which projection is trusted: `factor` x the image corner radius, cut at
    the first radius where the radial distortion r (1 + k1 r^2 + k2 r^4 + k3 r^6) stops increasing."""
    pw, ph = proj_size
    corners = np.array([[-0.5, -0.5, 1.0], [pw - 0.5, -0.5, 1.0], [-0.5, ph - 0.5, 1.0], [pw - 0.5, ph - 0.5, 1.0]])
    n = (np.linalg.inv(K_p) @ corners.T).T
    r_lim = factor * float(np.max(np.hypot(n[:, 0] / n[:, 2], n[:, 1] / n[:, 2])))
    k1, k2, _, _, k3 = dist
    r = np.linspace(0.0, r_lim, _RADIUS_SAMPLES)
    r2 = r * r
    rd = r * (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3)))
    falling = np.flatnonzero(np.diff(rd) <= 0.0)
    if falling.size:
        r_lim = float(r[falling[0]])
        logger.warning("Projector distortion folds back at normalised radius %.3f; projection is cut there", r_lim)
    return r_lim * r_lim


class DepthMeshProjector:
    """Render camera-space images into the projector frame through a per-pixel depth mesh (see module docstring)."""

    def __init__(self, cfg: dict, cam_size: tuple[int, int], proj_size: tuple[int, int],
                 K_cam: np.ndarray, K_p: np.ndarray, dist_p: np.ndarray, R: np.ndarray, t: np.ndarray) -> None:
        """`cfg` = config["render3d"]; sizes are (w, h); K_cam / K_p 3x3; dist_p OpenCV (k1, k2, p1, p2[, k3]);
        R (3x3), t (3,) map camera coords to projector coords: x_p = R X + t (metres)."""
        backend = cfg["backend"]
        if backend != "moderngl":
            raise ValueError(f"render3d.backend {backend!r} is not implemented; set it to 'moderngl' in config.yaml")
        self.depth_gap_m = float(cfg["depth_gap_m"])
        self.fill_holes_px = int(cfg["fill_holes_px"])
        self.mesh_step = int(cfg["mesh_step"])
        if self.mesh_step < 1 or self.fill_holes_px < 0 or not self.depth_gap_m > 0:
            raise ValueError(f"render3d needs mesh_step >= 1, fill_holes_px >= 0, depth_gap_m > 0; got {cfg}")
        self.cam_size = _size(cam_size)
        self.proj_size = _size(proj_size)
        K_cam = _mat3(K_cam, "K_cam")
        K_p = _mat3(K_p, "K_p")
        dist = _dist5(dist_p)
        factor = float(cfg.get("distortion_radius_factor", _DEFAULTS["distortion_radius_factor"]))
        r2_max = _trusted_r2(K_p, dist, self.proj_size, factor)

        w, h = self.cam_size
        pw, ph = self.proj_size
        quads_x, quads_y = (w - 1) // self.mesh_step, (h - 1) // self.mesh_step
        if quads_x < 1 or quads_y < 1:
            raise ValueError(f"mesh_step {self.mesh_step} leaves no grid quad in a {w}x{h} camera image")
        self._n_vertices = quads_x * quads_y * 6
        self.last_render_ms = 0.0
        self._out = np.zeros((ph, pw, 3), np.uint8)
        self._depth_buf = np.zeros((h, w), np.float32)
        self._valid_buf = np.zeros((h, w), np.bool_)

        try:
            self.ctx = moderngl.create_standalone_context(require=410)
        except Exception as e:  # noqa: BLE001 - re-raised with the fix-it message
            raise RuntimeError(f"render3d: cannot create an offscreen OpenGL 4.1 context ({e}). Fix: run on the Mac "
                               "with a GPU session (not over plain ssh), `pip install moderngl`") from e
        with self.ctx:
            ctx = self.ctx
            self._mesh_prog = ctx.program(vertex_shader=_MESH_VS, fragment_shader=_MESH_FS)
            self._dil_prog = ctx.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_DILATE_FS)
            self._res_prog = ctx.program(vertex_shader=_FULLSCREEN_VS, fragment_shader=_RESOLVE_FS)
            self._mesh_vao = ctx.vertex_array(self._mesh_prog, [])
            self._dil_vao = ctx.vertex_array(self._dil_prog, [])
            self._res_vao = ctx.vertex_array(self._res_prog, [])

            self._depth_tex = ctx.texture((w, h), 1, dtype="f4")
            self._depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._image_tex = ctx.texture((w, h), 3, dtype="f1")
            self._image_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._image_tex.repeat_x = self._image_tex.repeat_y = False
            self._mesh_tex = ctx.texture((pw, ph), 4, dtype="f1")
            self._dil_tex = ctx.texture((pw, ph), 4, dtype="f1")
            self._out_tex = ctx.texture((pw, ph), 3, dtype="f1")
            for tex in (self._mesh_tex, self._dil_tex, self._out_tex):
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._mesh_fbo = ctx.framebuffer([self._mesh_tex], ctx.depth_renderbuffer((pw, ph)))
            self._dil_fbo = ctx.framebuffer([self._dil_tex])
            self._out_fbo = ctx.framebuffer([self._out_tex])

            m = self._mesh_prog
            m["depth_tex"].value = _UNIT_DEPTH
            m["image_tex"].value = _UNIT_IMAGE
            m["step"].value = self.mesh_step
            m["quads_x"].value = quads_x
            m["cam_size"].value = (float(w), float(h))
            m["proj_size"].value = (float(pw), float(ph))
            m["Kc_inv"].write(np.linalg.inv(K_cam).T.astype("f4").tobytes())   # GLSL mat3 is column-major
            m["Kp"].write(K_p.T.astype("f4").tobytes())
            m["k12p12"].value = tuple(float(v) for v in dist[:4])
            m["k3"].value = float(dist[4])
            m["r2_max"].value = r2_max
            m["gap"].value = self.depth_gap_m
            radius = self.fill_holes_px // 2
            self._dil_prog["mesh_tex"].value = _UNIT_MESH
            self._dil_prog["radius"].value = radius
            self._res_prog["mesh_tex"].value = _UNIT_MESH
            self._res_prog["dil_tex"].value = _UNIT_DIL
            self._res_prog["radius"].value = radius
            self._fill = radius > 0
        self.set_extrinsics(R, t)

    def set_extrinsics(self, R: np.ndarray, t: np.ndarray) -> None:
        """Replace the camera -> projector pose: x_p = R X + t (R 3x3, t (3,) metres)."""
        R = _mat3(R, "R")
        t = np.asarray(t, np.float64).ravel()
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            raise ValueError(f"t must be 3 finite values, got {t}")
        with self.ctx:
            self._mesh_prog["R"].write(R.T.astype("f4").tobytes())
            self._mesh_prog["t"].value = tuple(float(v) for v in t)

    def render(self, image_cam: np.ndarray, depth_cam: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """Project `image_cam` ((H, W, 3) BGR uint8, camera space) through `depth_cam` ((H, W) metres, 0 = invalid,
        already resampled to camera resolution). `valid` ((H, W) bool or uint8, nonzero = usable) further masks the
        depth. Returns the (ph, pw, 3) BGR uint8 projector frame: an internal buffer reused on every call (copy it to
        keep it). Sets `last_render_ms` (upload + draw + readback, GPU-synchronous)."""
        t0 = time.perf_counter()
        w, h = self.cam_size
        if image_cam.shape != (h, w, 3) or image_cam.dtype != np.uint8:
            raise ValueError(f"image_cam must be ({h}, {w}, 3) uint8 BGR, got {image_cam.shape} {image_cam.dtype}")
        if depth_cam.shape != (h, w):
            raise ValueError(f"depth_cam must be ({h}, {w}) metres at camera resolution, got {depth_cam.shape}")
        if valid is not None:
            if valid.shape != (h, w):
                raise ValueError(f"valid must be ({h}, {w}), got {valid.shape}")
            np.not_equal(valid, 0, out=self._valid_buf)
            np.multiply(depth_cam, self._valid_buf, out=self._depth_buf)
            depth = self._depth_buf
        elif depth_cam.dtype == np.float32 and depth_cam.flags.c_contiguous:
            depth = depth_cam
        else:
            np.copyto(self._depth_buf, depth_cam)
            depth = self._depth_buf
        image = image_cam if image_cam.flags.c_contiguous else np.ascontiguousarray(image_cam)

        with self.ctx:
            ctx = self.ctx
            self._depth_tex.write(depth)
            self._image_tex.write(image)
            self._depth_tex.use(_UNIT_DEPTH)
            self._image_tex.use(_UNIT_IMAGE)
            self._mesh_fbo.use()
            self._mesh_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
            ctx.enable(moderngl.DEPTH_TEST)
            self._mesh_vao.render(moderngl.TRIANGLES, vertices=self._n_vertices)
            ctx.disable(moderngl.DEPTH_TEST)
            self._mesh_tex.use(_UNIT_MESH)
            if self._fill:
                self._dil_fbo.use()
                self._dil_vao.render(moderngl.TRIANGLES, vertices=3)
            self._dil_tex.use(_UNIT_DIL)
            self._out_fbo.use()
            self._res_vao.render(moderngl.TRIANGLES, vertices=3)
            self._out_fbo.read_into(self._out, components=3)
        self.last_render_ms = (time.perf_counter() - t0) * 1e3
        return self._out

    def release(self) -> None:
        """Free the GL context and everything in it (the instance is unusable afterwards)."""
        if getattr(self, "ctx", None) is not None:
            self.ctx.release()
            self.ctx = None
