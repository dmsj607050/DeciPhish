"""Run the static extraction rules on an offline URL and HTML snapshot."""

from .extraction.html_rules import extract_html
from .extraction.url_rules import extract_url

# Extraction metadata is not fed to the classifier.
SNAPSHOT_TIME = "1970-01-01T00:00:00Z"


class Snapshot:
    def __init__(self, content):
        self.content = content

    def read(self, _case_id, _resource):
        return self.content


def make_view(url: str, html: bytes, *, profile: str = "ablation"):
    """Extract a local page using the specified experiment's raw input profile."""
    if profile == "phreshphish":
        from .extraction.phreshphish import extract_view

        return extract_view(url, html)
    if profile != "ablation":
        raise ValueError(f"Unknown evidence profile: {profile}")
    case = {"case_id": "snapshot", "observed_url": url}
    url_items, url_coverage, _, _ = extract_url(case, None, SNAPSHOT_TIME)
    # The raw-input evidence pipeline keeps HTML extraction modality-local.
    html_case = {
        "case_id": "snapshot",
        "observed_url": "",
        "html_resource": {"resource_id": "html", "acquired_at": SNAPSHOT_TIME},
    }
    if html.strip():
        html_items, html_coverage, _, _ = extract_html(
            html_case, Snapshot(html), SNAPSHOT_TIME
        )
    else:
        html_items, html_coverage = [], "MISSING"
    return {
        "url": url_items,
        "html": html_items,
        "coverage": {"url": url_coverage, "html": html_coverage},
    }
