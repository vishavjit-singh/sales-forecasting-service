"""A tiny ASGI test client: calls the Starlette app directly (starlette's TestClient needs httpx)."""
import asyncio
import json
from urllib.parse import urlencode


def call(app, method: str, path: str, query: dict | None = None, body=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "root_path": "", "query_string": urlencode(query or {}).encode(),
             "headers": [(b"host", b"test"), (b"content-type", b"application/json"),
                         (b"content-length", str(len(data)).encode())],
             "client": ("test", 1), "server": ("test", 80)}
    sent = {"body": b""}
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": data, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return sent["status"], json.loads(sent["body"])
