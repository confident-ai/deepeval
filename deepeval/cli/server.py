from typing import Dict, Optional
import socketserver
import http.server
import threading
import json

from deepeval.telemetry import set_logged_in_with

LOGGED_IN_WITH = "logged_in_with"


class PairingResult:
    def __init__(self):
        self.api_key: Optional[str] = None
        self.email: Optional[str] = None
        self.event = threading.Event()

def start_server(
    pairing_code: str,
    port: int,
    prod_url: str,
    result: Optional[PairingResult] = None,
) -> None:

    class PairingHandler(http.server.SimpleHTTPRequestHandler):

        def log_message(self, format, *args):
            pass  # Suppress default logging

        def _set_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", prod_url)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def do_OPTIONS(self):
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data: Dict = json.loads(body)
            except json.JSONDecodeError:
                data = {}

            if self.path == f"/{LOGGED_IN_WITH}":
                api_key = data.get(LOGGED_IN_WITH)
                email = data.get("email")
                pairing_code_received = data.get("pairing_code")

                if api_key and pairing_code == pairing_code_received:
                    if email:
                        set_logged_in_with(email)
                    if result is not None:
                        result.api_key = api_key
                        result.email = email
                        result.event.set()

                    self.send_response(200)
                    self._set_cors_headers()
                    self.end_headers()
                    threading.Thread(
                        target=httpd.shutdown, daemon=True
                    ).start()
                    return

                self.send_response(400)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b"Invalid pairing code or data")

    with socketserver.TCPServer(("localhost", port), PairingHandler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        thread.join()
