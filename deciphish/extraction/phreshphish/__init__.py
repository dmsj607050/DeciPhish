"""Raw evidence profile used by the archived PhreshPhish inference path."""

def extract_view(url: str, html: bytes):
    from .extraction.html_rules import extract_html
    from .extraction.url_rules import extract_url

    stamp = "2026-11-13T00:00:00Z"
    case = {"case_id": "local-page", "observed_url": url,
            "html_resource": {"resource_id": "local-html", "acquired_at": stamp}}

    class MemoryStore:
        def read(self, case_id, resource):
            return html

    url_items, url_status, _, _ = extract_url(case, None, stamp)
    html_items, html_status, _, _ = extract_html(
        dict(case, observed_url=""), MemoryStore(), stamp
    )
    return {"coverage": {"url": url_status, "html": html_status},
            "url": url_items, "html": html_items}
