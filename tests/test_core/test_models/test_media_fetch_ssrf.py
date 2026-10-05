"""
Security regression tests for SSRF-safe remote media fetching.

`fetch_remote_media` is used by the multimodal model integrations to download an
image referenced by an (untrusted) `MLLMImage.url`. It must refuse internal /
link-local / metadata targets and must be resistant to DNS rebinding (validate
and then pin the connection to the validated IP rather than re-resolving).
"""

import socket

import pytest

from deepeval.models.media_fetch import (
    fetch_remote_media,
    MediaFetchError,
    _is_disallowed_ip,
)


class TestIpClassification:
    @pytest.mark.parametrize(
        "ip",
        [
            "169.254.169.254",  # cloud metadata / link-local
            "127.0.0.1",  # loopback
            "10.0.0.5",  # private
            "192.168.1.1",  # private
            "172.16.0.1",  # private
            "0.0.0.0",  # unspecified
            "::1",  # ipv6 loopback
            "fe80::1",  # ipv6 link-local
            "fc00::1",  # ipv6 unique-local
            "::ffff:169.254.169.254",  # ipv4-mapped metadata
            "100.64.0.1",  # CGNAT / shared address space
            "not-an-ip",
        ],
    )
    def test_disallowed(self, ip):
        assert _is_disallowed_ip(ip) is True

    @pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
    def test_allowed(self, ip):
        assert _is_disallowed_ip(ip) is False


class TestFetchBlocks:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:8080/admin",
            "http://localhost/x",
            "http://10.0.0.5/",
            "http://[::1]/",
            "file:///etc/passwd",
            "ftp://example.com/x",
            "gopher://127.0.0.1:6379/_INFO",
        ],
    )
    def test_unsafe_urls_are_refused(self, url):
        with pytest.raises(MediaFetchError):
            fetch_remote_media(url, connect_timeout=2, read_timeout=2)

    def test_dns_rebinding_is_blocked(self, monkeypatch):
        """A hostname that resolves to an internal IP must be refused, even if
        it looks like an ordinary public hostname."""
        real_getaddrinfo = socket.getaddrinfo

        def fake_getaddrinfo(host, port, *args, **kwargs):
            if host == "rebind.example":
                return [
                    (
                        socket.AF_INET,
                        socket.SOCK_STREAM,
                        6,
                        "",
                        ("169.254.169.254", port),
                    )
                ]
            return real_getaddrinfo(host, port, *args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        with pytest.raises(MediaFetchError, match="non-public address"):
            fetch_remote_media(
                "http://rebind.example/latest/meta-data/",
                connect_timeout=2,
                read_timeout=2,
            )

    def test_mixed_public_and_private_resolution_is_refused(self, monkeypatch):
        """If a host resolves to several addresses and any one is internal, the
        fetch must be refused (an attacker must not slip in one bad IP)."""
        real_getaddrinfo = socket.getaddrinfo

        def fake_getaddrinfo(host, port, *args, **kwargs):
            if host == "mixed.example":
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port)),
                    (
                        socket.AF_INET,
                        socket.SOCK_STREAM,
                        6,
                        "",
                        ("127.0.0.1", port),
                    ),
                ]
            return real_getaddrinfo(host, port, *args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        with pytest.raises(MediaFetchError):
            fetch_remote_media(
                "http://mixed.example/x", connect_timeout=2, read_timeout=2
            )
