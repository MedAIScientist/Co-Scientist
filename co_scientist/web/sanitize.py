"""HTML sanitizer for rendered markdown.

The UI accepts model-authored prose (hypotheses, reviews, final overview),
renders it via python-markdown, and injects the result with `| safe`. Without
sanitization a hostile excerpt — for example a prompt-injection payload pulled
from a PubMed abstract — could ship `<script>` or `<iframe>` straight to the
browser.

This module renders markdown then runs the HTML through a strict tag-allowlist
parser built on stdlib `html.parser`. It is intentionally narrow: no <img>
(would let a hostile excerpt phone home), no <a target=_blank> (tabnabbing
unless we also add rel=noopener), no <style>/<script>/<iframe>/<object>.
"""

from __future__ import annotations

import re
from html import escape as html_escape
from html.parser import HTMLParser
from urllib.parse import urlsplit

import markdown as _md

# Conservative allowlist. Markdown-rendered output rarely needs more than this.
_ALLOWED_TAGS: dict[str, frozenset[str]] = {
    "p": frozenset(),
    "br": frozenset(),
    "hr": frozenset(),
    "h1": frozenset(),
    "h2": frozenset(),
    "h3": frozenset(),
    "h4": frozenset(),
    "h5": frozenset(),
    "h6": frozenset(),
    "strong": frozenset(),
    "em": frozenset(),
    "b": frozenset(),
    "i": frozenset(),
    "u": frozenset(),
    "code": frozenset(),
    "pre": frozenset(),
    "blockquote": frozenset(),
    "ul": frozenset(),
    "ol": frozenset(),
    "li": frozenset(),
    "table": frozenset(),
    "thead": frozenset(),
    "tbody": frozenset(),
    "tr": frozenset(),
    "th": frozenset({"align"}),
    "td": frozenset({"align"}),
    "a": frozenset({"href", "title"}),
    "span": frozenset(),
    "div": frozenset(),
    "sup": frozenset(),
    "sub": frozenset(),
}

_VOID_TAGS = {"br", "hr"}

_SAFE_URL_SCHEMES = {"http", "https", "mailto"}


def _safe_href(value: str) -> str | None:
    """Allow http/https/mailto and relative URLs; reject everything else
    (including javascript:, data:, vbscript:, file:).
    """
    v = value.strip()
    if not v:
        return None
    # Relative URLs (no scheme) are fine — they cannot run JS.
    if "://" not in v and not v.startswith(("mailto:", "javascript:", "data:", "vbscript:", "file:")):
        return v
    scheme = urlsplit(v).scheme.lower()
    if scheme in _SAFE_URL_SCHEMES:
        return v
    return None


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._out: list[str] = []
        # When we drop a tag (e.g. <script>), we also drop its contents until
        # the matching end-tag. _drop_depth counts nesting.
        self._drop_tag: str | None = None
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._drop_depth and tag == self._drop_tag:
            self._drop_depth += 1
            return
        if tag in ("script", "style", "iframe", "object", "embed", "form", "link", "meta"):
            self._drop_tag = tag
            self._drop_depth = 1
            return
        if tag not in _ALLOWED_TAGS:
            return  # silently drop unknown tags but keep children
        out_attrs: list[str] = []
        allowed_attrs = _ALLOWED_TAGS[tag]
        for k, v in attrs:
            kl = k.lower()
            if kl.startswith("on"):  # event handlers
                continue
            if kl not in allowed_attrs:
                continue
            if kl == "href":
                if v is None:
                    continue
                safe = _safe_href(v)
                if safe is None:
                    continue
                v = safe
            if v is None:
                out_attrs.append(f'{kl}=""')
            else:
                out_attrs.append(f'{kl}="{html_escape(v, quote=True)}"')
        if tag == "a" and not any(a.startswith("rel=") for a in out_attrs):
            out_attrs.append('rel="noopener noreferrer nofollow"')
        attr_str = (" " + " ".join(out_attrs)) if out_attrs else ""
        if tag in _VOID_TAGS:
            self._out.append(f"<{tag}{attr_str} />")
        else:
            self._out.append(f"<{tag}{attr_str}>")

    def handle_endtag(self, tag: str) -> None:
        if self._drop_depth:
            if tag == self._drop_tag:
                self._drop_depth -= 1
                if self._drop_depth == 0:
                    self._drop_tag = None
            return
        if tag in _ALLOWED_TAGS and tag not in _VOID_TAGS:
            self._out.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # e.g. <br /> or <hr />
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS and tag in _ALLOWED_TAGS:
            self._out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._drop_depth:
            return
        self._out.append(html_escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        if self._drop_depth:
            return
        self._out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._drop_depth:
            return
        self._out.append(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self._out)


def sanitize_html(html: str) -> str:
    """Run a tag/attribute/scheme allowlist over `html`. Safe to inject as
    `| safe` in a Jinja2 template."""
    s = _Sanitizer()
    s.feed(html)
    s.close()
    return s.get_html()


# Extensions kept conservative: tables for evidence tables, fenced_code so
# tool excerpts render readably, sane_lists so bullets don't collapse.
_MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists"]

# Regex to strip the markdown library's "[TOC]" inline HTML blocks. We don't
# enable the toc extension, but cheap belt-and-suspenders against an
# attacker-controlled raw-HTML inline block being interpreted before we
# sanitize.
_RAW_HTML_BLOCK = re.compile(r"</?\s*(script|style|iframe|object|embed)[^>]*>", re.IGNORECASE)


def render_markdown(text: str) -> str:
    """Render `text` as markdown, then sanitize the HTML output.

    Always returns a string safe to render via `| safe` in templates.
    """
    if not text:
        return ""
    # Strip the most dangerous raw-HTML blocks pre-render. The sanitizer would
    # catch them anyway, but pre-stripping avoids weird markdown parsing
    # interactions (e.g. <script> in a code fence vs. raw).
    cleaned = _RAW_HTML_BLOCK.sub("", text)
    rendered = _md.markdown(cleaned, extensions=_MD_EXTENSIONS, output_format="html")
    return sanitize_html(rendered)
