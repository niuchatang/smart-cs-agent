"""Import administrative boundary GeoJSON into MySQL GIS tables.

Expected input is a GeoJSON FeatureCollection with Polygon or MultiPolygon
geometries in WGS84 lon/lat. The importer accepts common property names:
adcode/ad_code/code, name, level, parent_adcode, province, city, district.

Example:
  python scripts/import_admin_boundaries.py \
    --geojson /data/china_admin_boundaries.geojson \
    --schema database/weather_gis_schema.sql \
    --truncate
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pymysql  # type: ignore[import-untyped]
from dotenv import load_dotenv
from pymysql.cursors import DictCursor  # type: ignore[import-untyped]

load_dotenv()

Point = Tuple[float, float]
Ring = List[Point]
Polygon = List[Ring]


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.geojson).read_text(encoding="utf-8"))
    rows = normalize_rows(payload, include_levels=set(args.levels.split(",")) if args.levels else None)
    if not rows:
        raise SystemExit("No importable Polygon/MultiPolygon features found.")

    conn = connect(args)
    try:
        if args.schema:
            apply_schema(conn, Path(args.schema).read_text(encoding="utf-8"))
        if args.truncate:
            truncate_tables(conn, args.boundary_table, args.alias_table)
        inserted = insert_rows(conn, rows, args.boundary_table, args.alias_table, batch_size=args.batch_size)
        conn.commit()
    finally:
        conn.close()
    print(f"Imported {inserted} admin boundaries into {args.boundary_table}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", required=True, help="Path to admin boundary GeoJSON.")
    parser.add_argument("--schema", default="database/weather_gis_schema.sql", help="DDL file to apply before import.")
    parser.add_argument("--truncate", action="store_true", help="Clear GIS tables before import.")
    parser.add_argument("--levels", default="province,city,district", help="Comma-separated levels to import.")
    parser.add_argument("--boundary-table", default=os.getenv("GIS_BOUNDARY_TABLE", "weather_admin_boundary"))
    parser.add_argument("--alias-table", default=os.getenv("GIS_ALIAS_TABLE", "weather_admin_alias"))
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--host", default=os.getenv("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "3306") or 3306))
    parser.add_argument("--user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.getenv("MYSQL_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("MYSQL_DATABASE", "smart_cs_agent"))
    parser.add_argument("--unix-socket", default=os.getenv("MYSQL_UNIX_SOCKET", ""))
    return parser.parse_args()


def connect(args: argparse.Namespace) -> Any:
    kwargs: Dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
        "database": args.database,
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
    }
    if args.unix_socket and Path(args.unix_socket).exists():
        kwargs.pop("host", None)
        kwargs.pop("port", None)
        kwargs["unix_socket"] = args.unix_socket
    return pymysql.connect(**kwargs)


def apply_schema(conn: Any, schema_sql: str) -> None:
    cleaned_lines = []
    for line in schema_sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        cleaned_lines.append(line)
    statements = [s.strip() for s in "\n".join(cleaned_lines).split(";") if s.strip()]
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    conn.commit()


def truncate_tables(conn: Any, boundary_table: str, alias_table: str) -> None:
    boundary = safe_identifier(boundary_table)
    alias = safe_identifier(alias_table)
    with conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        cur.execute(f"TRUNCATE TABLE {alias}")
        cur.execute(f"TRUNCATE TABLE {boundary}")
        cur.execute("SET FOREIGN_KEY_CHECKS=1")
    conn.commit()


def normalize_rows(payload: Dict[str, Any], include_levels: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    features = list(iter_features(payload))
    raw_rows: List[Dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
        wkt = geometry_to_wkt(geom)
        if not wkt:
            continue
        adcode = first_str(props, "adcode", "ad_code", "code", "gb", "id")
        name = first_str(props, "name", "fullname", "full_name", "district", "city", "province")
        if not adcode or not name:
            continue
        level = normalize_level(first_str(props, "level", "type", "adcode_level"), adcode)
        if include_levels and level not in include_levels:
            continue
        center_lon, center_lat = centroid_from_wkt_source(geom)
        raw_rows.append(
            {
                "adcode": adcode,
                "name": name,
                "province": first_str(props, "province", "province_name"),
                "city": first_str(props, "city", "city_name", "prefecture"),
                "district": first_str(props, "district", "district_name", "county"),
                "level": level,
                "parent_adcode": first_str(props, "parent_adcode", "parent", "parent_code"),
                "center_lon": center_lon,
                "center_lat": center_lat,
                "wkt": wkt,
            }
        )

    fill_hierarchy(raw_rows)
    return raw_rows


def iter_features(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if payload.get("type") == "FeatureCollection":
        for item in payload.get("features") or []:
            if isinstance(item, dict):
                yield item
    elif payload.get("type") == "Feature":
        yield payload


def first_str(props: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = props.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_level(raw: str, adcode: str) -> str:
    text = (raw or "").strip().lower()
    mapping = {
        "province": "province",
        "prov": "province",
        "city": "city",
        "prefecture": "city",
        "district": "district",
        "county": "district",
        "area": "district",
    }
    if text in mapping:
        return mapping[text]
    code = re.sub(r"\D", "", adcode)
    if len(code) >= 6:
        if code.endswith("0000"):
            return "province"
        if code.endswith("00"):
            return "city"
        return "district"
    return "district"


def fill_hierarchy(rows: List[Dict[str, Any]]) -> None:
    by_code = {r["adcode"]: r for r in rows}
    province_by_prefix = {r["adcode"][:2]: r for r in rows if r["level"] == "province" and len(r["adcode"]) >= 2}
    city_by_prefix = {r["adcode"][:4]: r for r in rows if r["level"] == "city" and len(r["adcode"]) >= 4}

    for row in rows:
        code = row["adcode"]
        province = province_by_prefix.get(code[:2])
        city = city_by_prefix.get(code[:4])
        if row["level"] == "province":
            row["province"] = row["province"] or row["name"]
            row["city"] = row["city"] or (row["name"] if row["name"] in {"北京市", "上海市", "天津市", "重庆市"} else "")
        elif row["level"] == "city":
            row["province"] = row["province"] or (province["name"] if province else "")
            row["city"] = row["city"] or row["name"]
            row["parent_adcode"] = row["parent_adcode"] or (province["adcode"] if province else "")
        else:
            row["province"] = row["province"] or (province["name"] if province else "")
            row["city"] = row["city"] or (city["name"] if city else "")
            row["district"] = row["district"] or row["name"]
            row["parent_adcode"] = row["parent_adcode"] or (city["adcode"] if city else "")
            if not row["city"] and row["province"] in {"北京市", "上海市", "天津市", "重庆市"}:
                row["city"] = row["province"]
                row["parent_adcode"] = row["parent_adcode"] or row["province"][:2] + "0000"

        parent = by_code.get(row.get("parent_adcode", ""))
        if parent and not row["province"]:
            row["province"] = parent.get("province", "") or parent.get("name", "")


def geometry_to_wkt(geometry: Dict[str, Any]) -> str:
    gtype = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and isinstance(coords, list):
        polygons = [normalize_polygon(coords)]
    elif gtype == "MultiPolygon" and isinstance(coords, list):
        polygons = [normalize_polygon(poly) for poly in coords if isinstance(poly, list)]
    else:
        return ""
    polygons = [p for p in polygons if p]
    if not polygons:
        return ""
    return "MULTIPOLYGON(" + ",".join(polygon_to_wkt(poly) for poly in polygons) + ")"


def normalize_polygon(raw_polygon: Sequence[Any]) -> List[Ring]:
    polygon: List[Ring] = []
    for raw_ring in raw_polygon:
        if not isinstance(raw_ring, list):
            continue
        ring: Ring = []
        for point in raw_ring:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                ring.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        if len(ring) >= 3:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            polygon.append(ring)
    return polygon


def polygon_to_wkt(polygon: List[Ring]) -> str:
    # MySQL SRID 4326 geographic WKT uses latitude longitude order.
    rings = []
    for ring in polygon:
        pairs = ",".join(f"{lat:.8f} {lon:.8f}" for lon, lat in ring)
        rings.append(f"({pairs})")
    return "(" + ",".join(rings) + ")"


def centroid_from_wkt_source(geometry: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    points: List[Point] = []
    gtype = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and isinstance(coords, list):
        for ring in normalize_polygon(coords)[:1]:
            points.extend(ring)
    elif gtype == "MultiPolygon" and isinstance(coords, list):
        for poly in coords:
            if isinstance(poly, list):
                rings = normalize_polygon(poly)
                if rings:
                    points.extend(rings[0])
    if not points:
        return None, None
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def insert_rows(conn: Any, rows: List[Dict[str, Any]], boundary_table: str, alias_table: str, batch_size: int) -> int:
    boundary = safe_identifier(boundary_table)
    alias = safe_identifier(alias_table)
    inserted = 0
    with conn.cursor() as cur:
        for chunk in chunks(rows, max(1, batch_size)):
            boundary_values = [
                (
                    r["adcode"],
                    r["name"],
                    r["province"],
                    r["city"],
                    r["district"],
                    r["level"],
                    r["parent_adcode"],
                    r["center_lon"],
                    r["center_lat"],
                    r["wkt"],
                )
                for r in chunk
            ]
            cur.executemany(
                f"""
                INSERT INTO {boundary}
                  (adcode,name,province,city,district,level,parent_adcode,center_lon,center_lat,geom)
                VALUES
                  (%s,%s,%s,%s,%s,%s,%s,%s,%s,ST_GeomFromText(%s, 4326))
                ON DUPLICATE KEY UPDATE
                  name=VALUES(name),
                  province=VALUES(province),
                  city=VALUES(city),
                  district=VALUES(district),
                  level=VALUES(level),
                  parent_adcode=VALUES(parent_adcode),
                  center_lon=VALUES(center_lon),
                  center_lat=VALUES(center_lat),
                  geom=VALUES(geom)
                """,
                boundary_values,
            )
            alias_values = []
            for row in chunk:
                for alias_text, alias_type in build_aliases(row):
                    alias_values.append((row["adcode"], alias_text, alias_type))
            if alias_values:
                cur.executemany(
                    f"""
                    INSERT IGNORE INTO {alias}
                      (adcode,alias,alias_type)
                    VALUES (%s,%s,%s)
                    """,
                    alias_values,
                )
            inserted += len(chunk)
    return inserted


def build_aliases(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    candidates = [
        (row.get("adcode", ""), "adcode"),
        (row.get("name", ""), "name"),
        (row.get("province", ""), "province"),
        (row.get("city", ""), "city"),
        (row.get("district", ""), "district"),
        (f"{row.get('city', '')}{row.get('district', '')}" if row.get("city") and row.get("district") else "", "full_name"),
    ]
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for text, alias_type in candidates:
        t = str(text or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append((t, alias_type))
        if t.endswith(("市", "区", "县")) and len(t) > 2:
            short = t[:-1]
            if short not in seen:
                seen.add(short)
                out.append((short, "short_name"))
    return out


def chunks(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_identifier(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQL identifier: {value}")
    return value


if __name__ == "__main__":
    main()
