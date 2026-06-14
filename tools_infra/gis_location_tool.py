"""GIS location resolver for weather queries.

Resolution order:
1. MySQL spatial table, when GIS_ENABLE_MYSQL=true and table is available.
2. Local admin-boundary GeoJSON, when GIS_ADMIN_BOUNDARY_GEOJSON is configured.
3. Small built-in bbox set for development and examples.

Production deployments should provide nationwide city/district boundaries via
MySQL GIS or GeoJSON. The built-in bbox set is deliberately small and only
keeps local tests useful without large map data files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pymysql  # type: ignore[reportMissingImports]
from pymysql.cursors import DictCursor  # type: ignore[reportMissingImports]
from pydantic import BaseModel

from .location_parser import ParsedWeatherQuery
from .weather_cache import WeatherCache

Point = Tuple[float, float]  # lon, lat
Ring = List[Point]
Polygon = List[Ring]


class AdminAreaResult(BaseModel):
    ok: bool
    adcode: str = ""
    name: str = ""
    province: str = ""
    city: str = ""
    district: str = ""
    level: str = ""
    lon: Optional[float] = None
    lat: Optional[float] = None
    source: str = ""
    method: str = ""
    confidence: float = 0.0
    error: str = ""


_BUILTIN_BBOXES: List[Dict[str, Any]] = [
    {
        "adcode": "110000",
        "name": "北京市",
        "province": "北京市",
        "city": "北京市",
        "district": "",
        "level": "city",
        "bbox": (115.40, 39.40, 117.60, 41.10),
    },
    {
        "adcode": "110105",
        "name": "朝阳区",
        "province": "北京市",
        "city": "北京市",
        "district": "朝阳区",
        "level": "district",
        "bbox": (116.35, 39.78, 116.65, 40.12),
    },
    {
        "adcode": "110108",
        "name": "海淀区",
        "province": "北京市",
        "city": "北京市",
        "district": "海淀区",
        "level": "district",
        "bbox": (116.03, 39.88, 116.42, 40.18),
    },
    {
        "adcode": "120000",
        "name": "天津市",
        "province": "天津市",
        "city": "天津市",
        "district": "",
        "level": "city",
        "bbox": (116.70, 38.55, 118.05, 40.25),
    },
    {
        "adcode": "120101",
        "name": "和平区",
        "province": "天津市",
        "city": "天津市",
        "district": "和平区",
        "level": "district",
        "bbox": (117.17, 39.09, 117.23, 39.14),
    },
    {
        "adcode": "120102",
        "name": "河东区",
        "province": "天津市",
        "city": "天津市",
        "district": "河东区",
        "level": "district",
        "bbox": (117.19, 39.08, 117.32, 39.17),
    },
    {
        "adcode": "120103",
        "name": "河西区",
        "province": "天津市",
        "city": "天津市",
        "district": "河西区",
        "level": "district",
        "bbox": (117.16, 39.02, 117.28, 39.11),
    },
    {
        "adcode": "120104",
        "name": "南开区",
        "province": "天津市",
        "city": "天津市",
        "district": "南开区",
        "level": "district",
        "bbox": (117.09, 39.06, 117.20, 39.17),
    },
    {
        "adcode": "120105",
        "name": "河北区",
        "province": "天津市",
        "city": "天津市",
        "district": "河北区",
        "level": "district",
        "bbox": (117.17, 39.13, 117.27, 39.22),
    },
    {
        "adcode": "120106",
        "name": "红桥区",
        "province": "天津市",
        "city": "天津市",
        "district": "红桥区",
        "level": "district",
        "bbox": (117.08, 39.13, 117.20, 39.20),
    },
    {
        "adcode": "120110",
        "name": "东丽区",
        "province": "天津市",
        "city": "天津市",
        "district": "东丽区",
        "level": "district",
        "bbox": (117.25, 38.95, 117.60, 39.22),
    },
    {
        "adcode": "120111",
        "name": "西青区",
        "province": "天津市",
        "city": "天津市",
        "district": "西青区",
        "level": "district",
        "bbox": (116.85, 38.85, 117.18, 39.20),
    },
    {
        "adcode": "120112",
        "name": "津南区",
        "province": "天津市",
        "city": "天津市",
        "district": "津南区",
        "level": "district",
        "bbox": (117.18, 38.82, 117.55, 39.08),
    },
    {
        "adcode": "120113",
        "name": "北辰区",
        "province": "天津市",
        "city": "天津市",
        "district": "北辰区",
        "level": "district",
        "bbox": (117.02, 39.18, 117.35, 39.38),
    },
    {
        "adcode": "120114",
        "name": "武清区",
        "province": "天津市",
        "city": "天津市",
        "district": "武清区",
        "level": "district",
        "bbox": (116.78, 39.30, 117.35, 39.75),
    },
    {
        "adcode": "120115",
        "name": "宝坻区",
        "province": "天津市",
        "city": "天津市",
        "district": "宝坻区",
        "level": "district",
        "bbox": (117.10, 39.35, 117.75, 39.95),
    },
    {
        "adcode": "120116",
        "name": "滨海新区",
        "province": "天津市",
        "city": "天津市",
        "district": "滨海新区",
        "level": "district",
        "bbox": (117.35, 38.55, 118.05, 39.35),
    },
    {
        "adcode": "120117",
        "name": "宁河区",
        "province": "天津市",
        "city": "天津市",
        "district": "宁河区",
        "level": "district",
        "bbox": (117.45, 39.15, 117.95, 39.65),
    },
    {
        "adcode": "120118",
        "name": "静海区",
        "province": "天津市",
        "city": "天津市",
        "district": "静海区",
        "level": "district",
        "bbox": (116.70, 38.55, 117.30, 39.10),
    },
    {
        "adcode": "120119",
        "name": "蓟州区",
        "province": "天津市",
        "city": "天津市",
        "district": "蓟州区",
        "level": "district",
        "bbox": (117.15, 39.70, 117.80, 40.25),
    },
    {
        "adcode": "310000",
        "name": "上海市",
        "province": "上海市",
        "city": "上海市",
        "district": "",
        "level": "city",
        "bbox": (120.85, 30.65, 122.25, 31.90),
    },
    {
        "adcode": "310115",
        "name": "浦东新区",
        "province": "上海市",
        "city": "上海市",
        "district": "浦东新区",
        "level": "district",
        "bbox": (121.46, 30.80, 122.05, 31.42),
    },
    {
        "adcode": "330100",
        "name": "杭州市",
        "province": "浙江省",
        "city": "杭州市",
        "district": "",
        "level": "city",
        "bbox": (118.35, 29.15, 120.75, 30.65),
    },
    {
        "adcode": "330110",
        "name": "余杭区",
        "province": "浙江省",
        "city": "杭州市",
        "district": "余杭区",
        "level": "district",
        "bbox": (119.65, 30.20, 120.52, 30.65),
    },
    {
        "adcode": "440300",
        "name": "深圳市",
        "province": "广东省",
        "city": "深圳市",
        "district": "",
        "level": "city",
        "bbox": (113.75, 22.35, 114.65, 22.90),
    },
    {
        "adcode": "440305",
        "name": "南山区",
        "province": "广东省",
        "city": "深圳市",
        "district": "南山区",
        "level": "district",
        "bbox": (113.80, 22.38, 114.02, 22.63),
    },
]


class GISLocationTool:
    name = "GISLocationTool"

    def __init__(self, *, cache: WeatherCache | None = None, geojson_path: str | None = None) -> None:
        self.cache = cache or WeatherCache(ttl_seconds=600, namespace="gis")
        self.geojson_path = (geojson_path or os.getenv("GIS_ADMIN_BOUNDARY_GEOJSON", "")).strip()
        self.mysql_enabled = os.getenv("GIS_ENABLE_MYSQL", "").strip().lower() in {"1", "true", "yes", "on"}
        self.mysql_table = self._safe_identifier(
            os.getenv("GIS_BOUNDARY_TABLE", "weather_admin_boundary").strip() or "weather_admin_boundary",
            default="weather_admin_boundary",
        )
        self.mysql_alias_table = self._safe_identifier(
            os.getenv("GIS_ALIAS_TABLE", "weather_admin_alias").strip() or "weather_admin_alias",
            default="weather_admin_alias",
        )
        self._geojson_features: Optional[List[Dict[str, Any]]] = None

    def resolve_query(self, query: ParsedWeatherQuery, geocode_tool: Any | None = None) -> AdminAreaResult:
        preferred_level = "district" if query.district else ("city" if query.city and not query.district else "")
        lon = query.longitude
        lat = query.latitude
        if lon is not None and lat is not None:
            return self.resolve_point(float(lon), float(lat), preferred_level=preferred_level)

        name_hit = self.resolve_admin_name(
            query.location_text,
            city=query.city,
            district=query.district,
            preferred_level=preferred_level,
        )
        if name_hit.ok:
            return name_hit

        if geocode_tool is None or not query.location_text:
            return AdminAreaResult(ok=False, error="missing coordinate and geocode tool")

        geo = geocode_tool.geocode(query.location_text, city=query.city)
        if not getattr(geo, "ok", False):
            return AdminAreaResult(ok=False, error=getattr(geo, "error", "geocode failed"))
        lon = getattr(geo, "lon", None)
        lat = getattr(geo, "lat", None)
        if lon is None or lat is None:
            return self._from_geocode_fallback(geo)

        hit = self.resolve_point(float(lon), float(lat), preferred_level=preferred_level)
        if hit.ok:
            return hit
        return self._from_geocode_fallback(geo)

    def resolve_admin_name(
        self,
        name: str,
        *,
        city: str = "",
        district: str = "",
        preferred_level: str = "",
    ) -> AdminAreaResult:
        target = (district or name or "").strip()
        if not target:
            return AdminAreaResult(ok=False, error="missing admin name")
        key = f"gis-name:{preferred_level}:{city}:{district}:{target}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            return AdminAreaResult(**cached)

        for resolver in (self._resolve_admin_name_mysql, self._resolve_admin_name_geojson, self._resolve_admin_name_builtin):
            hit = resolver(target, city=city, preferred_level=preferred_level)
            if hit.ok:
                self.cache.set_json(key, hit.dict(), ttl_seconds=24 * 3600)
                return hit
        miss = AdminAreaResult(ok=False, error=f"admin name not in configured GIS boundaries: {target}")
        self.cache.set_json(key, miss.dict(), ttl_seconds=600)
        return miss

    def resolve_point(self, lon: float, lat: float, *, preferred_level: str = "") -> AdminAreaResult:
        key = f"gis-point:{preferred_level}:{lon:.6f}:{lat:.6f}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            return AdminAreaResult(**cached)

        for resolver in (self._resolve_point_mysql, self._resolve_point_geojson, self._resolve_point_builtin):
            hit = resolver(lon, lat, preferred_level=preferred_level)
            if hit.ok:
                self.cache.set_json(key, hit.dict(), ttl_seconds=24 * 3600)
                return hit

        miss = AdminAreaResult(ok=False, lon=lon, lat=lat, error="point not in configured GIS boundaries")
        self.cache.set_json(key, miss.dict(), ttl_seconds=600)
        return miss

    def _resolve_point_mysql(self, lon: float, lat: float, *, preferred_level: str = "") -> AdminAreaResult:
        if not self.mysql_enabled:
            return AdminAreaResult(ok=False, error="mysql gis disabled")
        conn = self._mysql_connect()
        if conn is None:
            return AdminAreaResult(ok=False, error="mysql gis connect failed")

        try:
            level_filter = ""
            args: List[Any] = [f"POINT({lat} {lon})"]
            if preferred_level in {"city", "district"}:
                level_filter = " AND level=%s"
                args.append(preferred_level)
            sql = (
                "SELECT adcode,name,province,city,district,level "
                f"FROM {self.mysql_table} "
                "WHERE ST_Contains(geom, ST_GeomFromText(%s, 4326))"
                f"{level_filter} "
                "ORDER BY CASE level WHEN 'district' THEN 1 WHEN 'city' THEN 2 ELSE 9 END "
                "LIMIT 1"
            )
            with conn.cursor() as cur:
                cur.execute(sql, args)
                row = cur.fetchone()
            if not row:
                return AdminAreaResult(ok=False, lon=lon, lat=lat, error="mysql gis no hit")
            return self._row_to_result(row, lon=lon, lat=lat, source="mysql", method="gis_mysql_st_contains")
        except Exception as exc:
            return AdminAreaResult(ok=False, lon=lon, lat=lat, error=f"mysql gis query failed: {exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _resolve_admin_name_mysql(self, name: str, *, city: str = "", preferred_level: str = "") -> AdminAreaResult:
        if not self.mysql_enabled:
            return AdminAreaResult(ok=False, error="mysql gis disabled")
        conn = self._mysql_connect()
        if conn is None:
            return AdminAreaResult(ok=False, error="mysql gis connect failed")
        try:
            aliases = self._name_candidates(name, city)
            level_filter = ""
            args: List[Any] = []
            if preferred_level in {"city", "district"}:
                level_filter = " AND b.level=%s"
                args.append(preferred_level)

            city_filter = ""
            if city:
                city_filter = " AND (b.city=%s OR b.province=%s OR b.name=%s)"
                args.extend([city, city, city])

            alias_marks = ",".join(["%s"] * len(aliases))
            sql = (
                "SELECT b.adcode,b.name,b.province,b.city,b.district,b.level "
                f"FROM {self.mysql_table} b "
                f"LEFT JOIN {self.mysql_alias_table} a ON a.adcode=b.adcode "
                f"WHERE (b.adcode IN ({alias_marks}) OR b.name IN ({alias_marks}) OR "
                f"b.district IN ({alias_marks}) OR b.city IN ({alias_marks}) OR a.alias IN ({alias_marks}))"
                f"{level_filter}{city_filter} "
                "ORDER BY CASE b.level WHEN 'district' THEN 1 WHEN 'city' THEN 2 ELSE 9 END, "
                "CHAR_LENGTH(b.name) ASC "
                "LIMIT 1"
            )
            all_alias_args = aliases * 5
            with conn.cursor() as cur:
                cur.execute(sql, all_alias_args + args)
                row = cur.fetchone()
            if not row:
                return AdminAreaResult(ok=False, error="mysql admin name no hit")
            return self._row_to_result(row, lon=None, lat=None, source="mysql", method="gis_mysql_admin_lookup")
        except Exception as exc:
            return AdminAreaResult(ok=False, error=f"mysql admin name query failed: {exc}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _mysql_connect(self) -> Any | None:
        try:
            kwargs: Dict[str, Any] = {
                "host": os.getenv("MYSQL_HOST", "127.0.0.1") or "127.0.0.1",
                "port": int(os.getenv("MYSQL_PORT", "3306") or 3306),
                "user": os.getenv("MYSQL_USER", "root") or "root",
                "password": os.getenv("MYSQL_PASSWORD", ""),
                "database": os.getenv("MYSQL_DATABASE", "smart_cs_agent") or "smart_cs_agent",
                "charset": "utf8mb4",
                "cursorclass": DictCursor,
                "connect_timeout": 1,
                "read_timeout": 2,
                "write_timeout": 2,
            }
            unix_socket = os.getenv("MYSQL_UNIX_SOCKET", "").strip()
            if unix_socket and Path(unix_socket).exists():
                kwargs.pop("host", None)
                kwargs.pop("port", None)
                kwargs["unix_socket"] = unix_socket
            return pymysql.connect(**kwargs)
        except Exception:
            return None

    @staticmethod
    def _safe_identifier(value: str, *, default: str) -> str:
        if not value.replace("_", "").isalnum():
            return default
        return value

    def _resolve_point_geojson(self, lon: float, lat: float, *, preferred_level: str = "") -> AdminAreaResult:
        features = self._load_geojson_features()
        if not features:
            return AdminAreaResult(ok=False, error="geojson boundary not configured")
        matches: List[AdminAreaResult] = []
        for feature in features:
            props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
            level = str(props.get("level") or props.get("adcode_level") or "").strip()
            if preferred_level and level and level != preferred_level:
                continue
            geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
            if self._geometry_contains(geom, lon, lat):
                matches.append(self._row_to_result(props, lon=lon, lat=lat, source="geojson", method="gis_geojson_point_in_polygon"))
        if not matches:
            return AdminAreaResult(ok=False, lon=lon, lat=lat, error="geojson gis no hit")
        matches.sort(key=lambda x: 0 if x.level == "district" else 1)
        return matches[0]

    def _resolve_admin_name_geojson(self, name: str, *, city: str = "", preferred_level: str = "") -> AdminAreaResult:
        features = self._load_geojson_features()
        if not features:
            return AdminAreaResult(ok=False, error="geojson boundary not configured")
        aliases = set(self._name_candidates(name, city))
        for feature in features:
            props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
            level = str(props.get("level") or props.get("adcode_level") or "").strip()
            if preferred_level and level and level != preferred_level:
                continue
            values = {
                str(props.get("adcode") or "").strip(),
                str(props.get("name") or "").strip(),
                str(props.get("district") or "").strip(),
                str(props.get("city") or "").strip(),
            }
            if aliases & values:
                return self._row_to_result(props, lon=None, lat=None, source="geojson", method="gis_geojson_admin_lookup")
        return AdminAreaResult(ok=False, error="geojson admin name no hit")

    def _resolve_point_builtin(self, lon: float, lat: float, *, preferred_level: str = "") -> AdminAreaResult:
        rows = self._ordered_builtin_rows(preferred_level)
        for row in rows:
            min_lon, min_lat, max_lon, max_lat = row["bbox"]
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                return self._row_to_result(row, lon=lon, lat=lat, source="builtin_bbox", method="gis_builtin_bbox")
        return AdminAreaResult(ok=False, lon=lon, lat=lat, error="builtin gis no hit")

    def _resolve_admin_name_builtin(self, name: str, *, city: str = "", preferred_level: str = "") -> AdminAreaResult:
        aliases = set(self._name_candidates(name, city))
        for row in self._ordered_builtin_rows(preferred_level):
            values = {
                str(row.get("adcode") or "").strip(),
                str(row.get("name") or "").strip(),
                str(row.get("district") or "").strip(),
                str(row.get("city") or "").strip(),
            }
            if aliases & values:
                return self._row_to_result(row, lon=None, lat=None, source="builtin_bbox", method="gis_builtin_admin_lookup")
        return AdminAreaResult(ok=False, error="builtin admin name no hit")

    @staticmethod
    def _ordered_builtin_rows(preferred_level: str) -> List[Dict[str, Any]]:
        rows = list(_BUILTIN_BBOXES)
        if preferred_level in {"city", "district"}:
            exact = [x for x in rows if x.get("level") == preferred_level]
            rest = [x for x in rows if x.get("level") != preferred_level]
            return exact + rest
        rows.sort(key=lambda x: 0 if x.get("level") == "district" else 1)
        return rows

    def _load_geojson_features(self) -> List[Dict[str, Any]]:
        if self._geojson_features is not None:
            return self._geojson_features
        path = self.geojson_path
        if not path:
            self._geojson_features = []
            return self._geojson_features
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("type") == "FeatureCollection" and isinstance(payload.get("features"), list):
                self._geojson_features = [f for f in payload["features"] if isinstance(f, dict)]
            elif payload.get("type") == "Feature":
                self._geojson_features = [payload]
            else:
                self._geojson_features = []
        except Exception:
            self._geojson_features = []
        return self._geojson_features

    @staticmethod
    def _geometry_contains(geometry: Dict[str, Any], lon: float, lat: float) -> bool:
        gtype = str(geometry.get("type") or "")
        coords = geometry.get("coordinates")
        if gtype == "Polygon" and isinstance(coords, list):
            return GISLocationTool._polygon_contains(GISLocationTool._normalize_polygon(coords), lon, lat)
        if gtype == "MultiPolygon" and isinstance(coords, list):
            for poly in coords:
                if isinstance(poly, list) and GISLocationTool._polygon_contains(GISLocationTool._normalize_polygon(poly), lon, lat):
                    return True
        return False

    @staticmethod
    def _normalize_polygon(raw: Sequence[Any]) -> Polygon:
        polygon: Polygon = []
        for raw_ring in raw:
            if not isinstance(raw_ring, list):
                continue
            ring: Ring = []
            for point in raw_ring:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        ring.append((float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        continue
            if ring:
                polygon.append(ring)
        return polygon

    @staticmethod
    def _polygon_contains(polygon: Polygon, lon: float, lat: float) -> bool:
        if not polygon:
            return False
        outer = polygon[0]
        if not GISLocationTool._ring_contains(outer, lon, lat):
            return False
        for hole in polygon[1:]:
            if GISLocationTool._ring_contains(hole, lon, lat):
                return False
        return True

    @staticmethod
    def _ring_contains(ring: Ring, lon: float, lat: float) -> bool:
        inside = False
        if len(ring) < 3:
            return False
        x, y = lon, lat
        j = len(ring) - 1
        for i, (xi, yi) in enumerate(ring):
            xj, yj = ring[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _row_to_result(row: Dict[str, Any], *, lon: Optional[float], lat: Optional[float], source: str, method: str) -> AdminAreaResult:
        name = str(row.get("name") or row.get("district") or row.get("city") or row.get("province") or "").strip()
        level = str(row.get("level") or ("district" if row.get("district") else "city")).strip()
        return AdminAreaResult(
            ok=True,
            adcode=str(row.get("adcode") or "").strip(),
            name=name,
            province=str(row.get("province") or "").strip(),
            city=str(row.get("city") or row.get("province") or "").strip(),
            district=str(row.get("district") or (name if level == "district" else "")).strip(),
            level=level,
            lon=lon,
            lat=lat,
            source=source,
            method=method,
            confidence=0.98 if source in {"mysql", "geojson"} else 0.75,
        )

    @staticmethod
    def _name_candidates(name: str, city: str = "") -> List[str]:
        raw = [name, f"{city}{name}" if city and not name.startswith(city) else ""]
        out: List[str] = []
        seen: set[str] = set()
        for item in raw:
            t = (item or "").strip()
            if not t:
                continue
            variants = {t}
            if t.endswith("市"):
                variants.add(t[:-1])
            else:
                variants.add(f"{t}市")
            if t.endswith("区") or t.endswith("县"):
                variants.add(t[:-1])
            for v in variants:
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
        return out

    @staticmethod
    def _from_geocode_fallback(geo: Any) -> AdminAreaResult:
        return AdminAreaResult(
            ok=bool(getattr(geo, "adcode", "")),
            adcode=str(getattr(geo, "adcode", "") or ""),
            name=str(getattr(geo, "district", "") or getattr(geo, "city", "") or getattr(geo, "province", "") or ""),
            province=str(getattr(geo, "province", "") or ""),
            city=str(getattr(geo, "city", "") or getattr(geo, "province", "") or ""),
            district=str(getattr(geo, "district", "") or ""),
            level="district" if getattr(geo, "district", "") else "city",
            lon=getattr(geo, "lon", None),
            lat=getattr(geo, "lat", None),
            source=str(getattr(geo, "source", "") or "geocode"),
            method="geocode_fallback",
            confidence=0.55,
            error="" if getattr(geo, "adcode", "") else "geocode missing adcode",
        )
