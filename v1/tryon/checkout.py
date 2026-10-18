"""Mock Visa checkout, shaped like a CyberSource /pts/v2/payments call.

submit_payment() is the one function that picks the processor. With
VISA_API_KEY, VISA_SHARED_SECRET and VISA_MERCHANT_ID in .env it calls the
CyberSource sandbox (apitest.cybersource.com) with HTTP Signature auth;
otherwise it posts the same request body to a local mock server that answers
like CyberSource. Payments run on a worker thread so rendering never waits, and
every attempt is appended to transactions.jsonl.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import queue
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from email.utils import formatdate
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAYMENTS_PATH = "/pts/v2/payments"


@dataclass
class Order:
    order_id: str
    garment_id: str
    name: str
    price: float
    currency: str


@dataclass
class PaymentResult:
    ok: bool
    status: str
    transaction_id: str
    message: str
    provider: str
    latency_ms: float
    raw: dict = field(default_factory=dict)


def load_env(path) -> dict:
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


def visa_credentials(cfg) -> dict | None:
    env = {**load_env(cfg.path(cfg.env_path)), **{k: v for k, v in os.environ.items() if k.startswith("VISA_")}}
    key, secret = env.get("VISA_API_KEY"), env.get("VISA_SHARED_SECRET")
    if not key or not secret:
        return None
    merchant = env.get("VISA_MERCHANT_ID")
    if not merchant:
        print("[checkout] VISA_API_KEY/VISA_SHARED_SECRET set but VISA_MERCHANT_ID missing; using the mock")
        return None
    return {"key": key, "secret": secret, "merchant": merchant, "host": env.get("VISA_HOST", "apitest.cybersource.com")}


def payment_body(order: Order) -> dict:
    """CyberSource authorization request using the public Visa sandbox test card."""
    return {
        "clientReferenceInformation": {"code": order.order_id},
        "processingInformation": {"capture": True},
        "paymentInformation": {"card": {"number": "4111111111111111", "expirationMonth": "12", "expirationYear": "2031"}},
        "orderInformation": {
            "amountDetails": {"totalAmount": f"{order.price:.2f}", "currency": order.currency},
            "billTo": {
                "firstName": "Demo", "lastName": "Shopper", "address1": "185 Stevens Way",
                "locality": "Seattle", "administrativeArea": "WA", "postalCode": "98195",
                "country": "US", "email": "demo@example.com",
            },
            "lineItems": [{"productName": order.name, "productSku": order.garment_id,
                           "unitPrice": f"{order.price:.2f}", "quantity": 1}],
        },
    }


def _parse(resp: dict, provider: str, latency_ms: float) -> PaymentResult:
    status = resp.get("status", "UNKNOWN")
    ok = status in ("AUTHORIZED", "PENDING", "TRANSMITTED")
    msg = resp.get("message") or resp.get("errorInformation", {}).get("message", "") or status
    return PaymentResult(ok, status, resp.get("id", ""), msg, provider, latency_ms, resp)


def _cybersource_request(body: bytes, creds: dict, target_label: str) -> urllib.request.Request:
    """HTTP Signature auth: HmacSHA256 over the listed headers with the base64-decoded secret."""
    digest = "SHA-256=" + base64.b64encode(hashlib.sha256(body).digest()).decode()
    date = formatdate(usegmt=True)
    signing = "\n".join([
        f"host: {creds['host']}",
        f"date: {date}",
        f"{target_label}: post {PAYMENTS_PATH}",
        f"digest: {digest}",
        f"v-c-merchant-id: {creds['merchant']}",
    ])
    sig = base64.b64encode(hmac.new(base64.b64decode(creds["secret"]), signing.encode(), hashlib.sha256).digest()).decode()
    headers = {
        "host": creds["host"], "date": date, "digest": digest, "v-c-merchant-id": creds["merchant"],
        "content-type": "application/json",
        "signature": (f'keyid="{creds["key"]}", algorithm="HmacSHA256", '
                      f'headers="host date {target_label} digest v-c-merchant-id", signature="{sig}"'),
    }
    return urllib.request.Request(f"https://{creds['host']}{PAYMENTS_PATH}", data=body, headers=headers, method="POST")


def cybersource_payment(order: Order, creds: dict) -> PaymentResult:
    body = json.dumps(payment_body(order)).encode()
    t0 = time.monotonic()
    resp: dict = {}
    # CyberSource's SDKs sign "(request-target)"; parts of its docs show "request-target".
    # Try the SDK form and fall back once if the signature is rejected.
    for label in ("(request-target)", "request-target"):
        try:
            with urllib.request.urlopen(_cybersource_request(body, creds, label), timeout=15) as r:
                resp = json.loads(r.read() or b"{}")
            break
        except urllib.error.HTTPError as e:
            resp = json.loads(e.read() or b"{}") if e.fp else {}
            resp.setdefault("status", f"HTTP_{e.code}")
            if e.code != 401:
                break
    return _parse(resp, "cybersource", (time.monotonic() - t0) * 1000)


class _MockVisaHandler(BaseHTTPRequestHandler):
    latency_s = 0.9
    decline_rate = 0.0

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        time.sleep(self.latency_s)
        amount = req.get("orderInformation", {}).get("amountDetails", {})
        declined = random.random() < self.decline_rate
        resp = {
            "id": str(int(time.time() * 1000)) + str(random.randint(100000, 999999)),
            "submitTimeUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "DECLINED" if declined else "AUTHORIZED",
            "clientReferenceInformation": req.get("clientReferenceInformation", {}),
            "orderInformation": {"amountDetails": {"authorizedAmount": amount.get("totalAmount"),
                                                   "currency": amount.get("currency")}},
            "processorInformation": {"approvalCode": f"{random.randint(0, 999999):06d}", "responseCode": "100"},
            "paymentAccountInformation": {"card": {"type": "001", "brand": "VISA"}},
        }
        if declined:
            resp["errorInformation"] = {"reason": "PROCESSOR_DECLINED", "message": "Decline - mock processor"}
        data = json.dumps(resp).encode()
        self.send_response(201 if not declined else 402)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class _LocalServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind does a reverse-DNS getfqdn(), which stalls for tens of seconds on venue wifi
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


_mock_lock = threading.Lock()
_mock_server: ThreadingHTTPServer | None = None


def _mock_url(cfg) -> str:
    global _mock_server
    with _mock_lock:
        if _mock_server is None:
            _MockVisaHandler.latency_s = cfg.checkout_mock_latency_ms / 1000.0
            _MockVisaHandler.decline_rate = cfg.checkout_mock_decline_rate
            _mock_server = _LocalServer(("127.0.0.1", 0), _MockVisaHandler)
            threading.Thread(target=_mock_server.serve_forever, name="mock-visa", daemon=True).start()
        return f"http://127.0.0.1:{_mock_server.server_address[1]}{PAYMENTS_PATH}"


def mock_payment(order: Order, cfg) -> PaymentResult:
    body = json.dumps(payment_body(order)).encode()
    t0 = time.monotonic()
    req = urllib.request.Request(_mock_url(cfg), data=body, headers={"content-type": "application/json"}, method="POST")
    # localhost: skip macOS system-proxy lookup, which can stall urlopen for tens of seconds
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=10) as r:
            resp = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        resp = json.loads(e.read() or b"{}")
    return _parse(resp, "mock", (time.monotonic() - t0) * 1000)


def submit_payment(order: Order, cfg) -> PaymentResult:
    """The single swap point between the mock and the real processor."""
    creds = visa_credentials(cfg)
    if creds:
        return cybersource_payment(order, creds)
    return mock_payment(order, cfg)


class CheckoutClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self._results: queue.Queue = queue.Queue()
        self.provider = "cybersource" if visa_credentials(cfg) else "mock"
        if self.provider == "mock":
            _mock_url(cfg)  # start the mock endpoint now, not on the first purchase

    def submit(self, garment) -> Order:
        order = Order(f"fitcast-{uuid.uuid4().hex[:10]}", garment.id, garment.name, garment.price, self.cfg.currency)
        threading.Thread(target=self._run, args=(order,), name="checkout", daemon=True).start()
        return order

    def _run(self, order: Order) -> None:
        try:
            result = submit_payment(order, self.cfg)
        except Exception as e:  # network down, bad credentials, malformed reply
            result = PaymentResult(False, "ERROR", "", str(e), self.provider, 0.0)
        self._log(order, result)
        self._results.put(result)

    def _log(self, order: Order, result: PaymentResult) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "order": asdict(order),
               "result": {k: v for k, v in asdict(result).items() if k != "raw"}, "response": result.raw}
        try:
            with open(self.cfg.path(self.cfg.transactions_path), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"[checkout] could not log transaction: {e}")

    def poll(self) -> PaymentResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None
