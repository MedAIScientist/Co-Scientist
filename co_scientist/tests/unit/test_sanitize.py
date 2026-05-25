"""Tests for the web markdown sanitizer."""

from __future__ import annotations

from co_scientist.web.sanitize import render_markdown, sanitize_html


def test_script_tag_is_stripped() -> None:
    out = sanitize_html("<p>hi</p><script>alert(1)</script>")
    assert "<script" not in out.lower()
    assert "alert" not in out
    assert "<p>hi</p>" in out


def test_event_handler_attribute_is_stripped() -> None:
    out = sanitize_html('<a href="https://x.com" onclick="alert(1)">x</a>')
    assert "onclick" not in out.lower()
    assert "alert" not in out


def test_javascript_url_is_dropped() -> None:
    out = sanitize_html('<a href="javascript:alert(1)">click</a>')
    assert "javascript" not in out.lower()


def test_data_url_is_dropped() -> None:
    out = sanitize_html('<a href="data:text/html,<script>alert(1)</script>">x</a>')
    assert "data:" not in out.lower()


def test_iframe_is_stripped_with_contents() -> None:
    out = sanitize_html("<iframe src='x'><p>inside</p></iframe>after")
    assert "iframe" not in out.lower()
    assert "inside" not in out
    assert "after" in out


def test_relative_href_is_kept() -> None:
    out = sanitize_html('<a href="/foo">bar</a>')
    assert 'href="/foo"' in out
    assert "rel=" in out  # noopener auto-added


def test_render_markdown_renders_basic_md() -> None:
    out = render_markdown("# title\n\n*hi*")
    assert "<h1>" in out
    assert "<em>" in out


def test_render_markdown_strips_raw_script_in_md() -> None:
    out = render_markdown("# title\n\n<script>alert(1)</script>\n\nbody")
    # Tags must be gone (so the browser cannot execute), but the inner text
    # becoming visible plaintext is acceptable and reveals injection attempts.
    assert "<script" not in out.lower()
    assert "</script" not in out.lower()
    assert "body" in out


def test_render_markdown_does_not_execute_link_xss() -> None:
    out = render_markdown('[click](javascript:alert(1))')
    assert "javascript" not in out.lower()
