"""The small HTTP API on port 8080: typed speech, touch, status, what it remembers and forgetting, face
recognition on/off, quiet hours, emergency stop and resume."""

from robot.firmware.common import (  # noqa: F401
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
    json,
    threading,
)
from pinch.mind.privacy import FORGETTABLE


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
            if self.path.startswith("/status"):
                return self._send(fw.status())
            if self.path.startswith("/memory"):                 # what it remembers (names and counts)
                return self._send(fw.mind.memory_summary())
            self._send({"error": "try /status or /memory"}, 404)

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
            elif self.path == "/forget":                     # {"what": "people"} ... or "person:<id>"
                what = str(data.get("what", ""))
                if what not in FORGETTABLE and not what.startswith("person:"):
                    return self._send({"error": f"what: one of {', '.join(FORGETTABLE)} or person:<id>"}, 400)
                fw.forget(what)
            elif self.path == "/settings" and isinstance(data.get("face_recognition"), bool):
                fw.set_face_recognition(data["face_recognition"])
            elif self.path == "/quiet_hours":                # {"start": 22, "end": 7} or {"off": true}
                if data.get("off"):
                    fw.set_quiet_hours(None)
                else:
                    try:
                        start, end = float(data["start"]), float(data["end"])
                    except (KeyError, TypeError, ValueError):
                        return self._send({"error": "give start and end hours (0-24), or off"}, 400)
                    if not (0 <= start < 24 and 0 <= end < 24) or start == end:
                        return self._send({"error": "start and end must be different hours in 0-24"}, 400)
                    fw.set_quiet_hours((start, end))
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
