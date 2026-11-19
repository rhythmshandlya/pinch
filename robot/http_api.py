"""The small HTTP API on port 8080: typed speech, touch, status, emergency stop and resume."""

from robot.firmware.common import (  # noqa: F401
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
    json,
    threading,
)


def serve(fw: "PinchFirmware", port=8080):
    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(fw.status() if self.path.startswith("/status") else {"error": "try /status"},
                       200 if self.path.startswith("/status") else 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send({"error": "bad json"}, 400)
            if self.path == "/say" and isinstance(data.get("text"), str):
                fw.say(data["text"][:300])
            elif self.path == "/touch" and data.get("gesture") in ("stroke", "hold", "tickle", "boop", "tap", "claw_hold"):
                fw.touch(data["gesture"])
            elif self.path == "/stop":                       # emergency stop: limp at once, until /resume
                fw.estop(str(data.get("why", "http"))[:80])
            elif self.path == "/resume":
                fw.resume()
            else:
                return self._send({"error": "unknown"}, 400)
            self._send({"ok": True})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
