# Fitcast — a projector that dresses you

Trying clothes on means finding them, carrying them to a fitting room, and changing
into each one. Fitcast skips all of that: **you stand in front of a projector in a
plain white shirt and white pants, pick a garment, and the projector paints it onto
your body, sized and positioned to fit you.** Swipe a hand to change items. Turn
around and the garment turns with you, so you see the front, then the side, then the
back.

DubHacks 2026, Visa track.

## How it works

1. **Calibrate.** The projector shows a known pattern and the camera finds it. That
   gives a mapping from camera pixels to projector pixels, so anything found in the
   camera image can be drawn back onto the same spot in the room.
2. **Track the body.** Pose estimation on the camera feed finds shoulders, hips,
   elbows and knees every frame.
3. **Read which way you face.** Shoulder width narrows as you turn sideways, and
   whether your face is visible tells front from back. Together they choose the
   front, side or back view of the garment.
4. **Fit the garment.** The chosen view is scaled and warped to your shoulders and
   hips (shirts) or hips and knees (pants), then mapped into projector space and
   drawn.
5. **Swipe.** A hand moving quickly across the body switches to the next or previous
   item. Shirts and pants are separate slots.

The white clothes are the projection screen: they make the colors read true and the
garment's edges look like real fabric edges.

## Demo scope

- **Must have:** one shirt and one pair of pants, each with front, side and back
  views, tracking one person live.
- **Nice to have:** a small catalog to swipe through, and smooth blending between
  views while turning.
- **Stretch:** any product, from its catalog photos alone.

## Setup

- A projector, and a webcam mounted next to it, both aimed at the same spot.
- A dim room, so the projection beats the ambient light.
- A baggy white T-shirt and white pants.

## Status

Repository created; build in progress.
