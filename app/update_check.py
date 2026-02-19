"""
In-app update check: compare current version with latest release (GitHub or JSON URL).
Prompt users to download when a newer version exists. Configure via env vars or constants:
- UPDATE_CHECK_GITHUB_REPO: "owner/repo" for GitHub Releases (e.g. "YourOrg/rapid-scribe").
- UPDATE_CHECK_JSON_URL: URL returning {"version": "3.1", "url": "https://..."} (overrides GitHub).
When changing __version__, also update installer.iss: AppVersion and OutputBaseFilename.
"""
import json
import os
import re
import urllib.request
from typing import Optional

# For a public GitHub repo: set to "owner/repo".
# Override with env var UPDATE_CHECK_GITHUB_REPO so you don't have to edit code.
# Private repos are not supported here (API requires auth); use UPDATE_CHECK_JSON_URL instead.
GITHUB_REPO = os.environ.get("UPDATE_CHECK_GITHUB_REPO", "").strip() or "aj-ebels/meetings"

# Optional: JSON URL that returns {"version": "3.1", "url": "https://..."}.
# Override with env var UPDATE_CHECK_JSON_URL. If set, GitHub check is skipped.
UPDATE_CHECK_JSON_URL = os.environ.get("UPDATE_CHECK_JSON_URL", "").strip() or None

# Request timeout (seconds)
REQUEST_TIMEOUT = 10


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Normalize version string to a tuple of integers for comparison.
    'v3.1.0' -> (3, 1, 0), '3.0' -> (3, 0).
    """
    if not version_str or not isinstance(version_str, str):
        return (0,)
    s = version_str.strip().lower()
    if s.startswith("v"):
        s = s[1:]
    parts = re.split(r"[.\-]", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Take only leading digits
        match = re.match(r"^(\d+)", p)
        if match:
            out.append(int(match.group(1)))
        else:
            break
    return tuple(out) if out else (0,)


def is_newer_version(current: str, latest: str) -> bool:
    """Return True if latest is newer than current (e.g. 3.1 > 3.0)."""
    cur = _parse_version(current)
    lat = _parse_version(latest)
    for i in range(max(len(cur), len(lat))):
        c = cur[i] if i < len(cur) else 0
        l = lat[i] if i < len(lat) else 0
        if l > c:
            return True
        if l < c:
            return False
    return False


def _fetch_github_latest(owner: str, repo: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch latest release from GitHub API. Returns (version_str, download_url, error)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, None, "No releases found."
        return None, None, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, None, str(e.reason) if getattr(e, "reason", None) else str(e)
    except (json.JSONDecodeError, OSError) as e:
        return None, None, str(e)

    tag = data.get("tag_name") or ""
    version_str = tag.strip()
    if version_str.lower().startswith("v"):
        version_str = version_str[1:].strip()

    # Prefer first .exe asset for Windows installer, else first asset, else release page
    download_url = data.get("html_url") or ""
    assets = data.get("assets") or []
    for a in assets:
        name = (a.get("name") or "").lower()
        u = a.get("browser_download_url")
        if u and name.endswith(".exe"):
            download_url = u
            break
    if not download_url and assets:
        u = assets[0].get("browser_download_url")
        if u:
            download_url = u

    return version_str or None, download_url or None, None


def _fetch_json_url(url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch version and download URL from a JSON endpoint. Returns (version_str, download_url, error)."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return None, None, str(e.reason) if getattr(e, "reason", None) else str(e)
    except (json.JSONDecodeError, OSError) as e:
        return None, None, str(e)

    if not isinstance(data, dict):
        return None, None, "Invalid response format"
    version_str = data.get("version")
    if version_str is None:
        return None, None, "Missing 'version' in response"
    version_str = str(version_str).strip()
    download_url = (data.get("url") or data.get("download_url") or "").strip()
    return version_str, download_url or None, None


def check_for_updates(
    current_version: str,
    *,
    github_repo: Optional[str] = None,
    json_url: Optional[str] = None,
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Check for a newer version. Returns (has_update, latest_version, download_url, error).

    Uses json_url if provided, else github_repo (e.g. "owner/repo"), else module defaults
    (UPDATE_CHECK_JSON_URL env, then GITHUB_REPO / UPDATE_CHECK_GITHUB_REPO).
    """
    json_url = json_url or UPDATE_CHECK_JSON_URL
    github_repo = github_repo or GITHUB_REPO

    if json_url:
        version_str, download_url, err = _fetch_json_url(json_url)
    elif github_repo:
        parts = github_repo.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return False, None, None, "Invalid github_repo (use owner/repo)"
        version_str, download_url, err = _fetch_github_latest(parts[0].strip(), parts[1].strip())
    else:
        return False, None, None, "No update source configured (set GITHUB_REPO or UPDATE_CHECK_JSON_URL)"

    if err:
        return False, None, None, err
    if not version_str:
        return False, None, None, "Could not read version"

    has_update = is_newer_version(current_version, version_str)
    return has_update, version_str, download_url, None
