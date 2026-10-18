import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import time
import unittest
import urllib.request
from unittest import mock

from tryon import checkout as co
from tryon.checkout import CheckoutClient, Order, PaymentResult, visa_credentials
from tryon.config import load_config
from tryon.garments import load_catalog
from tryon.state import (CHARGING, FAILED, IDLE, PROCESSING, SUCCESS, AppState)

_CATALOG = None


def catalog(cfg):
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog(cfg)
    return _CATALOG


def ok_result():
    return PaymentResult(True, "AUTHORIZED", "123", "ok", "mock", 1.0)


def bad_result():
    return PaymentResult(False, "DECLINED", "", "no", "mock", 1.0)


class AppStateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(overrides={"checkout_display_ms": 1000})
        self.cat = catalog(self.cfg)
        self.st = AppState(self.cfg, self.cat)

    def test_initial(self):
        self.assertEqual(self.st.mode, "shirt")
        self.assertEqual(self.st.checkout, IDLE)
        self.assertIsNotNone(self.st.active)
        self.assertEqual(self.st.active.id, self.cat.items("shirt")[0].id)

    def test_step_wraps_both_ways(self):
        n = len(self.cat.items("shirt"))
        self.assertGreaterEqual(n, 2)
        ids = [self.st.active.id]
        for _ in range(n):
            self.st.step(+1, 0.0)
            ids.append(self.st.active.id)
        self.assertEqual(ids[0], ids[-1])
        self.assertEqual(len(set(ids[:-1])), n)
        self.st.step(-1, 0.0)
        self.assertEqual(self.st.active.id, ids[n - 1])

    def test_step_sets_toast_that_expires(self):
        self.st.step(+1, 10.0)
        self.assertIn(self.st.active.name, self.st.toast.text)
        self.assertIn("$", self.st.toast.text)
        self.st.tick(10.5)
        self.assertIsNotNone(self.st.toast)
        self.st.tick(11.3)
        self.assertIsNone(self.st.toast)

    def test_toggle_mode(self):
        self.st.toggle_mode(0.0)
        self.assertEqual(self.st.mode, "pants")
        self.assertEqual(self.st.toast.text, "PANTS MODE")
        self.st.step(+1, 0.0)
        self.assertEqual(self.st.index["pants"], 1)
        self.assertEqual(self.st.index["shirt"], 0)
        self.st.toggle_mode(0.0)
        self.assertEqual(self.st.mode, "shirt")

    def test_toggle_mode_without_other_category(self):
        class Cat:
            def items(self, c):
                return self_cat.items(c) if c == "shirt" else []
        self_cat = self.cat
        st = AppState(self.cfg, Cat())
        st.toggle_mode(0.0)
        self.assertEqual(st.mode, "shirt")
        self.assertIn("no pants", st.toast.text)

    def test_charging(self):
        self.st.set_charging(0.4)
        self.assertEqual((self.st.checkout, self.st.checkout_progress), (CHARGING, 0.4))
        self.st.set_charging(0.0)
        self.assertEqual(self.st.checkout, IDLE)

    def test_begin_checkout(self):
        g = self.st.begin_checkout(0.0)
        self.assertIs(g, self.st.active)
        self.assertEqual(self.st.checkout, PROCESSING)
        self.assertEqual(self.st.checkout_progress, 1.0)
        self.assertFalse(self.st.checkout_ready)
        self.assertIsNone(self.st.begin_checkout(0.1))   # no double charge
        self.st.set_charging(0.5)                         # ignored while processing
        self.assertEqual(self.st.checkout, PROCESSING)

    def test_step_blocked_while_processing_and_success(self):
        self.st.begin_checkout(0.0)
        idx = dict(self.st.index)
        self.st.step(+1, 0.0)
        self.assertEqual(self.st.index, idx)
        self.st.finish_checkout(ok_result(), 1.0)
        self.st.step(+1, 1.0)
        self.assertEqual(self.st.index, idx)

    def test_step_allowed_after_failure(self):
        self.st.begin_checkout(0.0)
        self.st.finish_checkout(bad_result(), 1.0)
        self.assertEqual(self.st.checkout, FAILED)
        self.st.step(+1, 1.0)
        self.assertEqual(self.st.index["shirt"], 1)

    def test_finish_success_and_tick(self):
        g = self.st.begin_checkout(5.0)
        self.st.finish_checkout(ok_result(), 6.0)
        self.assertEqual(self.st.checkout, SUCCESS)
        self.assertEqual(self.st.checkout_garment, g)
        self.assertAlmostEqual(self.st.checkout_until, 7.0)
        self.st.tick(6.5)
        self.assertEqual(self.st.checkout, SUCCESS)
        self.st.tick(7.0)
        self.assertEqual(self.st.checkout, IDLE)
        self.assertEqual(self.st.checkout_progress, 0.0)
        self.assertTrue(self.st.checkout_ready)

    def test_finish_failure_and_tick(self):
        self.st.begin_checkout(0.0)
        self.st.finish_checkout(bad_result(), 0.0)
        self.assertEqual(self.st.checkout, FAILED)
        self.st.tick(1.5)
        self.assertEqual(self.st.checkout, IDLE)

    def test_finish_ignored_when_not_processing(self):
        self.st.finish_checkout(ok_result(), 0.0)
        self.assertEqual(self.st.checkout, IDLE)
        self.assertIsNone(self.st.checkout_result)

    def test_checkout_ready_after_failed_state(self):
        """FAILED is not in (IDLE, CHARGING) so a re-attempt is blocked until the receipt expires."""
        self.st.begin_checkout(0.0)
        self.st.finish_checkout(bad_result(), 0.0)
        self.assertFalse(self.st.checkout_ready)


class CheckoutClientTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fitcast-test-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.tx = os.path.join(self.dir, "tx.jsonl")
        self.env = os.path.join(self.dir, "none.env")
        # mock server singleton is process-wide: reset so config (latency/decline) is picked up
        co._mock_server = None
        co._MockVisaHandler.latency_s = 0.05
        co._MockVisaHandler.decline_rate = 0.0

    def tearDown(self):
        srv = co._mock_server
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        co._mock_server = None

    def cfg(self, **ov):
        base = {"transactions_path": self.tx, "env_path": self.env, "checkout_mock_latency_ms": 50.0}
        base.update(ov)
        return load_config(overrides=base)

    def wait(self, client, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            r = client.poll()
            if r is not None:
                return r
            time.sleep(0.02)
        self.fail("no checkout result within %.1fs" % timeout)

    def lines(self):
        with open(self.tx) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_end_to_end_authorized(self):
        cfg = self.cfg()
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in [k for k in os.environ if k.startswith("VISA_")]:
                os.environ.pop(k)
            client = CheckoutClient(cfg)
            self.assertEqual(client.provider, "mock")
            g = catalog(cfg).items("shirt")[0]
            self.assertIsNone(client.poll())
            order = client.submit(g)
            r = self.wait(client)
        self.assertTrue(r.ok)
        self.assertEqual(r.status, "AUTHORIZED")
        self.assertEqual(r.provider, "mock")
        self.assertTrue(r.transaction_id)
        self.assertEqual(order.garment_id, g.id)
        recs = self.lines()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["order"]["order_id"], order.order_id)
        self.assertEqual(recs[0]["result"]["status"], "AUTHORIZED")
        self.assertEqual(recs[0]["response"]["orderInformation"]["amountDetails"]["authorizedAmount"], f"{g.price:.2f}")
        self.assertNotIn("raw", recs[0]["result"])

    def test_two_orders_two_lines(self):
        cfg = self.cfg()
        client = CheckoutClient(cfg)
        g = catalog(cfg).items("shirt")[0]
        o1, o2 = client.submit(g), client.submit(g)
        self.assertNotEqual(o1.order_id, o2.order_id)
        self.wait(client)
        self.wait(client)
        self.assertEqual(len(self.lines()), 2)

    def test_decline_path(self):
        cfg = self.cfg(checkout_mock_decline_rate=1.0)
        client = CheckoutClient(cfg)
        g = catalog(cfg).items("shirt")[0]
        client.submit(g)
        r = self.wait(client)
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "DECLINED")
        self.assertIn("Decline", r.message)
        st = AppState(cfg, catalog(cfg))
        st.begin_checkout(0.0)
        st.finish_checkout(r, 0.0)
        self.assertEqual(st.checkout, FAILED)
        self.assertEqual(self.lines()[0]["result"]["ok"], False)

    def test_unreachable_processor_gives_error_result(self):
        cfg = self.cfg()
        client = CheckoutClient(cfg)
        g = catalog(cfg).items("shirt")[0]
        with mock.patch.object(co, "mock_payment", side_effect=OSError("boom")):
            client.submit(g)
            r = self.wait(client)
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "ERROR")
        self.assertEqual(len(self.lines()), 1)

    # -- credentials --------------------------------------------------------------
    def _clean_env(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("VISA_")}
        return mock.patch.dict(os.environ, env, clear=True)

    def test_visa_credentials_none_without_env(self):
        with self._clean_env():
            self.assertIsNone(visa_credentials(self.cfg()))

    def test_visa_credentials_from_env_file(self):
        p = os.path.join(self.dir, "t.env")
        with open(p, "w") as f:
            f.write('# comment\nVISA_API_KEY=key123\nexport VISA_SHARED_SECRET="c2VjcmV0"\nVISA_MERCHANT_ID = merch\n')
        with self._clean_env():
            c = visa_credentials(self.cfg(env_path=p))
        self.assertEqual(c["key"], "key123")
        self.assertEqual(c["secret"], "c2VjcmV0")
        self.assertEqual(c["merchant"], "merch")
        self.assertEqual(c["host"], "apitest.cybersource.com")

    def test_visa_credentials_missing_merchant_falls_back(self):
        p = os.path.join(self.dir, "t.env")
        with open(p, "w") as f:
            f.write("VISA_API_KEY=k\nVISA_SHARED_SECRET=c2Vj\n")
        with self._clean_env():
            self.assertIsNone(visa_credentials(self.cfg(env_path=p)))

    def test_visa_credentials_process_env_overrides_file(self):
        p = os.path.join(self.dir, "t.env")
        with open(p, "w") as f:
            f.write("VISA_API_KEY=k\nVISA_SHARED_SECRET=c2Vj\nVISA_MERCHANT_ID=m\n")
        with self._clean_env():
            os.environ["VISA_API_KEY"] = "envkey"
            self.assertEqual(visa_credentials(self.cfg(env_path=p))["key"], "envkey")

    def test_client_provider_cybersource_with_creds(self):
        p = os.path.join(self.dir, "t.env")
        with open(p, "w") as f:
            f.write("VISA_API_KEY=k\nVISA_SHARED_SECRET=c2Vj\nVISA_MERCHANT_ID=m\n")
        with self._clean_env():
            self.assertEqual(CheckoutClient(self.cfg(env_path=p)).provider, "cybersource")

    # -- signature ------------------------------------------------------------------
    def test_cybersource_signature_headers(self):
        secret_raw = b"super-secret-bytes!"
        creds = {"key": "KEYID", "secret": base64.b64encode(secret_raw).decode(), "merchant": "mymerch",
                 "host": "apitest.cybersource.com"}
        order = Order("fitcast-abc", "shirt_01", "Crimson Tee", 24.0, "USD")
        captured = {}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps({"id": "99", "status": "AUTHORIZED"}).encode()

        def fake_urlopen(req, timeout=None, **kw):
            captured["req"] = req
            return FakeResp()

        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            r = co.cybersource_payment(order, creds)
        self.assertTrue(r.ok)
        self.assertEqual(r.provider, "cybersource")
        req = captured["req"]
        self.assertEqual(req.full_url, "https://apitest.cybersource.com/pts/v2/payments")
        self.assertEqual(req.get_method(), "POST")
        h = {k.lower(): v for k, v in req.header_items()}
        body = req.data
        self.assertEqual(json.loads(body)["clientReferenceInformation"]["code"], "fitcast-abc")
        self.assertEqual(h["digest"], "SHA-256=" + base64.b64encode(hashlib.sha256(body).digest()).decode())
        self.assertEqual(h["v-c-merchant-id"], "mymerch")
        self.assertEqual(h["host"], "apitest.cybersource.com")
        self.assertEqual(h["content-type"], "application/json")
        self.assertRegex(h["date"], r"^\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} GMT$")
        m = re.fullmatch(r'keyid="KEYID", algorithm="HmacSHA256", headers="host date \(request-target\) digest v-c-merchant-id", '
                         r'signature="([A-Za-z0-9+/=]+)"', h["signature"])
        self.assertIsNotNone(m, h["signature"])
        signing = "\n".join([
            "host: apitest.cybersource.com", f"date: {h['date']}", "(request-target): post /pts/v2/payments",
            f"digest: {h['digest']}", "v-c-merchant-id: mymerch"])
        expect = base64.b64encode(hmac.new(secret_raw, signing.encode(), hashlib.sha256).digest()).decode()
        self.assertEqual(m.group(1), expect)

    def test_cybersource_signing_string_uses_target_per_spec(self):
        """The first attempt signs '(request-target)' as CyberSource's SDKs do."""
        creds = {"key": "K", "secret": base64.b64encode(b"s").decode(), "merchant": "m", "host": "apitest.cybersource.com"}
        cap = {}

        class R:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def read(s): return b"{}"
        with mock.patch.object(urllib.request, "urlopen", lambda req, timeout=None: (cap.update(req=req), R())[1]):
            co.cybersource_payment(Order("o", "g", "n", 1.0, "USD"), creds)
        sig = {k.lower(): v for k, v in cap["req"].header_items()}["signature"]
        self.assertIn('headers="host date (request-target) digest v-c-merchant-id"', sig)

    def test_cybersource_http_error_parsed(self):
        import io
        import urllib.error
        creds = {"key": "K", "secret": base64.b64encode(b"s").decode(), "merchant": "m", "host": "h.example"}

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "unauth", {}, io.BytesIO(b'{"message":"Authentication Failed"}'))
        with mock.patch.object(urllib.request, "urlopen", boom):
            r = co.cybersource_payment(Order("o", "g", "n", 1.0, "USD"), creds)
        self.assertFalse(r.ok)
        self.assertEqual(r.status, "HTTP_401")

    def test_cybersource_retries_other_target_label_on_401(self):
        import io
        import urllib.error
        creds = {"key": "K", "secret": base64.b64encode(b"s").decode(), "merchant": "m", "host": "h.example"}
        seen = []

        class R:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def read(s): return json.dumps({"id": "7", "status": "AUTHORIZED"}).encode()

        def first_rejects(req, timeout=None):
            sig = {k.lower(): v for k, v in req.header_items()}["signature"]
            seen.append(sig)
            if "(request-target)" in sig:
                raise urllib.error.HTTPError(req.full_url, 401, "unauth", {}, io.BytesIO(b"{}"))
            return R()
        with mock.patch.object(urllib.request, "urlopen", first_rejects):
            r = co.cybersource_payment(Order("o", "g", "n", 1.0, "USD"), creds)
        self.assertTrue(r.ok)
        self.assertEqual(len(seen), 2)
        self.assertIn('headers="host date request-target digest', seen[1])

    def test_payment_body_amount_format(self):
        b = co.payment_body(Order("o", "g", "n", 24, "USD"))
        self.assertEqual(b["orderInformation"]["amountDetails"]["totalAmount"], "24.00")


if __name__ == "__main__":
    unittest.main()
