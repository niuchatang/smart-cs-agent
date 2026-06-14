"""Download nationwide China admin boundaries from Aliyun DataV and merge to GeoJSON.

Data source: https://geo.datav.aliyun.com/areas_v3/bound/
License: check DataV terms before production use.

Example:
  .venv/bin/python scripts/download_admin_boundaries.py \
    --output data/china_admin_boundaries.geojson
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

DATAV_ROOT = "https://geo.datav.aliyun.com/areas_v3/bound"
DEFAULT_OUTPUT = "data/china_admin_boundaries.geojson"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching province list from {DATAV_ROOT}/100000_full.json ...")
    root = fetch_json(f"{DATAV_ROOT}/100000_full.json")
    provinces = root.get("features") or []
    adcodes = []
    for feat in provinces:
        props = feat.get("properties") or {}
        code = normalize_adcode(props.get("adcode"))
        if code:
            adcodes.append(code)

    if args.limit:
        adcodes = adcodes[: args.limit]
        print(f"Limiting to first {args.limit} provinces (dev mode).")

    all_features: List[Dict[str, Any]] = []
    for i, adcode in enumerate(adcodes, start=1):
        url = f"{DATAV_ROOT}/{adcode}_full.json"
        print(f"[{i}/{len(adcodes)}] {url}")
        try:
            payload = fetch_json(url)
        except Exception as exc:
            print(f"  skip {adcode}: {exc}")
            continue
        for feat in payload.get("features") or []:
            normalized = normalize_feature(feat)
            if normalized:
                all_features.append(normalized)
        if args.sleep:
            time.sleep(args.sleep)

    merged = {"type": "FeatureCollection", "features": all_features}
    output.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_features)} features -> {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Only download first N provinces (for testing).")
    parser.add_argument("--sleep", type=float, default=0.15, help="Seconds between province requests.")
    return parser.parse_args()


def fetch_json(url: str) -> Dict[str, Any]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON from {url}")
    return data


def normalize_feature(feature: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(feature, dict):
        return None
    props = dict(feature.get("properties") or {})
    geom = feature.get("geometry")
    if not isinstance(geom, dict):
        return None

    adcode = normalize_adcode(props.get("adcode"))
    name = str(props.get("name") or "").strip()
    if not adcode or not name:
        return None

    level = str(props.get("level") or infer_level(adcode)).strip().lower()
    parent = props.get("parent") if isinstance(props.get("parent"), dict) else {}
    parent_adcode = normalize_adcode(parent.get("adcode"))

    province = ""
    city = ""
    district = ""
    if level == "province":
        province = name
    elif level == "city":
        city = name
    else:
        district = name

    return {
        "type": "Feature",
        "properties": {
            "adcode": adcode,
            "name": name,
            "level": level,
            "parent_adcode": parent_adcode,
            "province": province,
            "city": city,
            "district": district,
        },
        "geometry": geom,
    }


def normalize_adcode(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    if len(digits) == 6:
        return digits
    if len(digits) < 6:
        return digits.ljust(6, "0")
    return digits[:6]


def infer_level(adcode: str) -> str:
    if adcode.endswith("0000"):
        return "province"
    if adcode.endswith("00"):
        return "city"
    return "district"


if __name__ == "__main__":
    main()
