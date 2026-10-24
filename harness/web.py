"""Public HTTP fetches with DNS validation and connections pinned to checked IPs."""

from datetime import datetime, timezone
import http.client
import ipaddress
import socket
import ssl
import time
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from markdownify import markdownify


def public_addresses(url: str) -> tuple:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only HTTP(S) URLs without credentials are allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    answers = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise PermissionError("Local, private, reserved and mixed public/private DNS targets are blocked")
    return parsed, addresses, port


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, hostname, address, port, timeout):
        super().__init__(hostname, port, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


class SafeFetcher:
    def __init__(self, max_bytes=1_000_000, timeout=10, redirects=3):
        self.max_bytes, self.timeout, self.redirects = max_bytes, timeout, redirects

    def fetch(self, url):
        deadline = time.monotonic() + self.timeout
        for attempt in range(self.redirects + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Fetch deadline exceeded")
            parsed, addresses, port = public_addresses(url)
            connection = (PinnedHTTPS(parsed.hostname, addresses[0], port, remaining)
                          if parsed.scheme == "https" else
                          http.client.HTTPConnection(addresses[0], port, timeout=remaining))
            host = parsed.hostname if port in {80, 443} else f"{parsed.hostname}:{port}"
            try:
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                connection.request("GET", path, headers={"Host": host, "Accept-Encoding": "identity",
                                                        "User-Agent": "Harness/3"})
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    target = response.getheader("Location")
                    if not target or attempt == self.redirects:
                        raise ValueError("Redirect limit reached or missing redirect target")
                    url = urljoin(url, target)
                    continue  # Resolve and validate the next host too.
                if not 200 <= response.status < 300:
                    raise ValueError(f"HTTP {response.status}")
                if response.getheader("Content-Encoding", "identity") != "identity":
                    raise ValueError("Compressed responses are not accepted")
                kind = response.getheader("Content-Type", "text/plain").split(";")[0].lower()
                if kind not in {"text/html", "text/plain", "text/markdown", "application/json"}:
                    raise ValueError(f"Unsupported content type: {kind}")
                raw = bytearray()
                while len(raw) <= self.max_bytes:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Fetch deadline exceeded")
                    if getattr(connection, "sock", None):
                        connection.sock.settimeout(remaining)
                    read = getattr(response, "read1", response.read)
                    chunk = read(min(65536, self.max_bytes + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                truncated = len(raw) > self.max_bytes
                text = raw[:self.max_bytes].decode("utf-8", errors="replace")
                if kind == "text/html":
                    soup = BeautifulSoup(text, "html.parser")
                    for element in soup(["script", "style", "noscript", "nav", "footer"]):
                        element.decompose()
                    text = markdownify(str(soup), heading_style="ATX")
                return {"url": url, "content": text, "truncated": truncated,
                        "captured_at": datetime.now(timezone.utc).isoformat()}
            finally:
                connection.close()
        raise ValueError("Redirect limit reached")
