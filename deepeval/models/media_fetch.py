"""
SSRF-safe fetching of remote media (images) referenced by multimodal test cases.

A multimodal ``MLLMImage`` can carry an ``http(s)://`` URL that originates from
untrusted data (a dataset, a traced message, a retrieved document). Fetching it
naively with ``requests.get(url)`` is a Server-Side Request Forgery vector: the
URL can point at internal/link-local services or the cloud metadata endpoint
(``169.254.169.254``) and the response is folded back into the model payload.

This module fetches such URLs safely:

* only ``http``/``https`` schemes are allowed;
* the host is resolved and **every** resolved address must be publicly routable
  (private, loopback, link-local, CGNAT, reserved, multicast and unspecified
  ranges are refused);
* the connection is **pinned to the validated IP** rather than re-resolved by
  the HTTP client. This is what prevents DNS-rebinding: without pinning an
  attacker's DNS can return a public IP for the validation lookup and a private
  IP for the actual request. TLS SNI / certificate validation still use the
  original hostname;
* redirects are followed manually and each hop is re-validated and re-pinned.
"""

import ipaddress
import socket
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import certifi
import urllib3

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_MAX_REDIRECTS = 3


class MediaFetchError(ValueError):
    """Raised when a remote media URL is unsafe to fetch or cannot be fetched."""


def _is_disallowed_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    # Normalise IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254) to its IPv4 form.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (
        not addr.is_global
        or addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _resolve_public_ips(host: str, port: int) -> List[str]:
    try:
        infos = socket.getaddrinfo(
            host, port, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as e:
        raise MediaFetchError(f"could not resolve host {host!r}: {e}")

    ips: List[str] = []
    for info in infos:
        ip = info[4][0].split("%")[0]  # drop any IPv6 scope id
        if ip not in ips:
            ips.append(ip)

    if not ips:
        raise MediaFetchError(f"host {host!r} did not resolve to any address")

    # Every resolved address must be public: an attacker must not be able to
    # slip in a single internal address alongside public ones.
    for ip in ips:
        if _is_disallowed_ip(ip):
            raise MediaFetchError(
                f"refusing to fetch {host!r}: resolves to non-public address {ip}"
            )
    return ips


def _host_header(host: str, port: int, scheme: str) -> str:
    bracketed = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    return bracketed if port == default_port else f"{bracketed}:{port}"


def fetch_remote_media(
    url: str,
    *,
    connect_timeout: Optional[float] = None,
    read_timeout: Optional[float] = None,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
) -> Tuple[bytes, Optional[str]]:
    """Fetch a remote media URL with SSRF (and DNS-rebinding) protection.

    Returns ``(content_bytes, content_type)``. Raises ``MediaFetchError`` if the
    URL is unsafe, cannot be resolved to a public address, or fails to fetch.
    """
    current = url
    for _ in range(max_redirects + 1):
        parsed = urlparse(current)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise MediaFetchError(
                f"unsupported URL scheme {scheme!r}; only http/https are allowed"
            )
        host = parsed.hostname
        if not host:
            raise MediaFetchError(f"URL has no host: {current!r}")
        port = parsed.port or (443 if scheme == "https" else 80)

        # Resolve + validate, then pin the connection to a validated IP so the
        # HTTP client cannot re-resolve to a different (internal) address.
        pinned_ip = _resolve_public_ips(host, port)[0]
        timeout = urllib3.Timeout(connect=connect_timeout, read=read_timeout)

        if scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                host=pinned_ip,
                port=port,
                cert_reqs="CERT_REQUIRED",
                ca_certs=certifi.where(),
                server_hostname=host,  # SNI + certificate hostname
                assert_hostname=host,
                timeout=timeout,
                retries=False,
                maxsize=1,
            )
        else:
            pool = urllib3.HTTPConnectionPool(
                host=pinned_ip,
                port=port,
                timeout=timeout,
                retries=False,
                maxsize=1,
            )

        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"

        try:
            resp = pool.urlopen(
                "GET",
                target,
                headers={"Host": _host_header(host, port, scheme)},
                redirect=False,
                preload_content=True,
                decode_content=False,
            )
            status = resp.status
            if status in _REDIRECT_STATUSES:
                location = resp.headers.get("Location")
                if not location:
                    raise MediaFetchError(
                        f"redirect with no Location while fetching {url!r}"
                    )
                current = urljoin(current, location)
                continue  # re-validate + re-pin the next hop
            if status >= 400:
                raise MediaFetchError(
                    f"failed to fetch {url!r}: HTTP status {status}"
                )
            return resp.data, resp.headers.get("Content-Type")
        except urllib3.exceptions.HTTPError as e:
            raise MediaFetchError(f"failed to fetch {url!r}: {e}")
        finally:
            pool.close()

    raise MediaFetchError(f"too many redirects while fetching {url!r}")
