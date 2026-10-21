#!/usr/bin/env python3
"""Fetch a few CC-licensed interior photos from Wikimedia Commons as local test input.

Why not scrape listings: ToS and copyright exposure, and it makes results unreproducible.
These are ordinary photographs of rooms — not listing photography — but they are real images
with citable licenses, which is what a local experiment needs.

    python scripts/fetch_samples.py --out samples --count 10

Writes `SOURCES.md` beside the images with title, license, author and page URL for each.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "pic-to-3d-poc/0.1 (local experiment)"}
QUERIES = [
    "living room interior", "kitchen interior apartment", "bedroom interior",
    "bathroom interior", "dining room interior", "empty apartment room",
]
# museum/artwork titles are not photographs and produce nonsense geometry
ARTWORK = re.compile(r"\b(painting|oil on|watercolou?r|c\.\s*1[6-9]\d\d|1[6-9]\d\d|museum|"
                     r"museum|google art|by [A-Z][a-z]+ [A-Z][a-z]+ -)", re.I)


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def strip_tags(s: str | None) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(s or "")).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--min-width", type=int, default=1200)
    ap.add_argument("--thumb-width", type=int, default=1400)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: set[str] = set()

    for query in QUERIES:
        if len(rows) >= args.count:
            break
        try:
            data = api({"action": "query", "generator": "search",
                        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
                        "gsrlimit": 12, "prop": "imageinfo",
                        "iiprop": "url|size|extmetadata",
                        "iiurlwidth": args.thumb_width})
        except Exception as exc:  # network flake: keep going
            print(f"query failed ({query}): {exc}")
            continue

        for page in data.get("query", {}).get("pages", {}).values():
            if len(rows) >= args.count:
                break
            info = page["imageinfo"][0]
            title = page["title"][5:]
            if page["title"] in seen:
                continue
            if info["width"] < args.min_width or info["width"] / info["height"] < 1.1:
                continue
            if ARTWORK.search(title):
                continue
            meta = info.get("extmetadata", {})
            lic = strip_tags(meta.get("LicenseShortName", {}).get("value", ""))
            if not lic or "fair" in lic.lower() or "non-free" in lic.lower():
                continue

            name = f"sample{len(rows) + 1:02d}.jpg"
            try:
                req = urllib.request.Request(info["thumburl"], headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    (out / name).write_bytes(r.read())
            except Exception as exc:
                print(f"download failed ({name}): {exc}")
                continue

            seen.add(page["title"])
            rows.append({
                "file": name, "title": title, "license": lic,
                "artist": strip_tags(meta.get("Artist", {}).get("value", ""))[:120],
                "page": "https://commons.wikimedia.org/wiki/"
                        + urllib.parse.quote(page["title"].replace(" ", "_")),
                "width": info["width"], "height": info["height"],
            })
            print(f"  {name}  {lic:<14} {title[:56]}")

    lines = ["# Sample photos", "",
             "CC-licensed images from Wikimedia Commons, downloaded by",
             "`scripts/fetch_samples.py`. Local test input only; attribution below.", ""]
    for r in rows:
        lines += [f"## {r['file']}", f"- title: {r['title']}", f"- license: {r['license']}",
                  f"- author: {r['artist'] or 'unknown'}", f"- source: {r['page']}",
                  f"- size: {r['width']}x{r['height']}", ""]
    (out / "SOURCES.md").write_text("\n".join(lines))
    print(f"\ndownloaded {len(rows)} photos -> {out} (see SOURCES.md for attribution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
