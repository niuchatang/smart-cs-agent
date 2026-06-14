"""Geocoding and weather tools backed by AMap with cache support."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional

import requests
from pydantic import BaseModel

from .gis_location_tool import GISLocationTool
from .location_parser import ParsedWeatherQuery
from .weather_cache import WeatherCache

HttpGet = Callable[..., requests.Response]


class GeoCodeResult(BaseModel):
    ok: bool
    query: str = ""
    adcode: str = ""
    province: str = ""
    city: str = ""
    district: str = ""
    formatted_address: str = ""
    location: str = ""
    lon: Optional[float] = None
    lat: Optional[float] = None
    source: str = "amap"
    error: str = ""


_BUILTIN_GEOCODES: Dict[str, Dict[str, Any]] = {
    "北京": {"adcode": "110000", "province": "北京市", "city": "北京市", "district": "", "lon": 116.4074, "lat": 39.9042},
    "北京市": {"adcode": "110000", "province": "北京市", "city": "北京市", "district": "", "lon": 116.4074, "lat": 39.9042},
    "北京市朝阳区": {"adcode": "110105", "province": "北京市", "city": "北京市", "district": "朝阳区", "lon": 116.4431, "lat": 39.9219},
    "朝阳区": {"adcode": "110105", "province": "北京市", "city": "北京市", "district": "朝阳区", "lon": 116.4431, "lat": 39.9219},
    "北京市海淀区": {"adcode": "110108", "province": "北京市", "city": "北京市", "district": "海淀区", "lon": 116.2981, "lat": 39.9599},
    "海淀区": {"adcode": "110108", "province": "北京市", "city": "北京市", "district": "海淀区", "lon": 116.2981, "lat": 39.9599},
    "上海": {"adcode": "310000", "province": "上海市", "city": "上海市", "district": "", "lon": 121.4737, "lat": 31.2304},
    "上海市": {"adcode": "310000", "province": "上海市", "city": "上海市", "district": "", "lon": 121.4737, "lat": 31.2304},
    "上海浦东新区": {"adcode": "310115", "province": "上海市", "city": "上海市", "district": "浦东新区", "lon": 121.5447, "lat": 31.2222},
    "上海市浦东新区": {"adcode": "310115", "province": "上海市", "city": "上海市", "district": "浦东新区", "lon": 121.5447, "lat": 31.2222},
    "浦东新区": {"adcode": "310115", "province": "上海市", "city": "上海市", "district": "浦东新区", "lon": 121.5447, "lat": 31.2222},
    "杭州": {"adcode": "330100", "province": "浙江省", "city": "杭州市", "district": "", "lon": 120.1551, "lat": 30.2741},
    "杭州市": {"adcode": "330100", "province": "浙江省", "city": "杭州市", "district": "", "lon": 120.1551, "lat": 30.2741},
    "杭州余杭区": {"adcode": "330110", "province": "浙江省", "city": "杭州市", "district": "余杭区", "lon": 120.2994, "lat": 30.4190},
    "杭州市余杭区": {"adcode": "330110", "province": "浙江省", "city": "杭州市", "district": "余杭区", "lon": 120.2994, "lat": 30.4190},
    "余杭区": {"adcode": "330110", "province": "浙江省", "city": "杭州市", "district": "余杭区", "lon": 120.2994, "lat": 30.4190},
    "深圳": {"adcode": "440300", "province": "广东省", "city": "深圳市", "district": "", "lon": 114.0579, "lat": 22.5431},
    "深圳市": {"adcode": "440300", "province": "广东省", "city": "深圳市", "district": "", "lon": 114.0579, "lat": 22.5431},
    "深圳南山区": {"adcode": "440305", "province": "广东省", "city": "深圳市", "district": "南山区", "lon": 113.9304, "lat": 22.5333},
    "深圳市南山区": {"adcode": "440305", "province": "广东省", "city": "深圳市", "district": "南山区", "lon": 113.9304, "lat": 22.5333},
    "南山区": {"adcode": "440305", "province": "广东省", "city": "深圳市", "district": "南山区", "lon": 113.9304, "lat": 22.5333},
}


class GeoCodeTool:
    name = "GeoCodeTool"

    def __init__(
        self,
        *,
        api_key: str = "",
        cache: WeatherCache | None = None,
        http_get: HttpGet | None = None,
        bypass_proxy: bool = True,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.cache = cache or WeatherCache(ttl_seconds=600)
        self.http_get = http_get
        self.bypass_proxy = bypass_proxy

    def geocode(self, address: str, *, city: str = "") -> GeoCodeResult:
        query = (address or "").strip()
        if not query:
            return GeoCodeResult(ok=False, query=query, error="missing address")
        key = f"geocode:{city}:{query}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            return GeoCodeResult(**cached)

        builtin = self._lookup_builtin(query)
        if builtin is not None:
            self.cache.set_json(key, builtin.dict(), ttl_seconds=24 * 3600)
            return builtin

        if not self.api_key:
            return GeoCodeResult(ok=False, query=query, error="AMAP_API_KEY is not configured")

        try:
            params = {"key": self.api_key, "address": query}
            if city:
                params["city"] = city
            resp = self._get("https://restapi.amap.com/v3/geocode/geo", params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("status", "0")) != "1":
                return GeoCodeResult(ok=False, query=query, error=str(payload.get("info", "amap geocode failed")))
            geocodes = payload.get("geocodes", [])
            if not isinstance(geocodes, list) or not geocodes:
                return GeoCodeResult(ok=False, query=query, error="geocode empty")
            row = geocodes[0] if isinstance(geocodes[0], dict) else {}
            location = str(row.get("location", "")).strip()
            lon = lat = None
            if "," in location:
                lon_s, lat_s = location.split(",", 1)
                lon, lat = float(lon_s), float(lat_s)
            result = GeoCodeResult(
                ok=True,
                query=query,
                adcode=str(row.get("adcode", "") or "").strip(),
                province=str(row.get("province", "") or "").strip(),
                city=str(row.get("city", "") or row.get("province", "") or "").strip(),
                district=str(row.get("district", "") or "").strip(),
                formatted_address=str(row.get("formatted_address", "") or query).strip(),
                location=location,
                lon=lon,
                lat=lat,
                source="amap",
            )
            self.cache.set_json(key, result.dict(), ttl_seconds=24 * 3600)
            return result
        except Exception as exc:
            return GeoCodeResult(ok=False, query=query, error=str(exc))

    @staticmethod
    def _lookup_builtin(query: str) -> GeoCodeResult | None:
        row = _BUILTIN_GEOCODES.get(query)
        if row is None:
            return None
        location = f"{row['lon']},{row['lat']}"
        return GeoCodeResult(
            ok=True,
            query=query,
            adcode=str(row["adcode"]),
            province=str(row.get("province", "")),
            city=str(row.get("city", "")),
            district=str(row.get("district", "")),
            formatted_address=query,
            location=location,
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            source="builtin",
        )

    def _get(self, url: str, *, params: Dict[str, Any], timeout: int) -> requests.Response:
        if self.http_get is not None:
            return self.http_get(url, params=params, timeout=timeout, bypass_proxy=self.bypass_proxy)
        if not self.bypass_proxy:
            return requests.get(url, params=params, timeout=timeout)
        with requests.Session() as session:
            session.trust_env = False
            return session.get(url, params=params, timeout=timeout)


class WeatherTool:
    name = "WeatherTool"

    def __init__(
        self,
        *,
        api_key: str = "",
        cache: WeatherCache | None = None,
        geocode_tool: GeoCodeTool | None = None,
        gis_tool: GISLocationTool | None = None,
        http_get: HttpGet | None = None,
        bypass_proxy: bool = True,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.cache = cache or WeatherCache(ttl_seconds=600)
        self.geocode_tool = geocode_tool or GeoCodeTool(api_key=self.api_key, cache=self.cache, http_get=http_get, bypass_proxy=bypass_proxy)
        self.gis_tool = gis_tool or GISLocationTool(cache=self.cache)
        self.http_get = http_get
        self.bypass_proxy = bypass_proxy
        self.aqi_url = os.getenv("AMAP_AQI_API_URL", "").strip()
        self.warning_url = os.getenv("AMAP_WARNING_API_URL", "").strip()

    def query(self, parsed: ParsedWeatherQuery | Dict[str, Any]) -> Dict[str, Any]:
        q = parsed if isinstance(parsed, ParsedWeatherQuery) else ParsedWeatherQuery(**parsed)
        if not q.location_text and (q.longitude is None or q.latitude is None):
            return {"city": "", "ok": False, "error": "missing weather location"}

        gis = self.gis_tool.resolve_query(q, geocode_tool=self.geocode_tool)
        if gis.ok:
            geo = GeoCodeResult(
                ok=True,
                query=q.location_text,
                adcode=gis.adcode,
                province=gis.province,
                city=gis.city,
                district=gis.district,
                formatted_address=gis.name or q.location_text,
                location=f"{gis.lon},{gis.lat}" if gis.lon is not None and gis.lat is not None else "",
                lon=gis.lon,
                lat=gis.lat,
                source=gis.source,
            )
        else:
            geo = self.geocode_tool.geocode(q.location_text, city=q.city)

        if not geo.ok:
            return {
                "city": q.location_text,
                "ok": False,
                "source": "amap_weather",
                "query": q.dict(),
                "gis": gis.dict(),
                "resolution_method": gis.method or "geocode",
                "error": geo.error or gis.error or "geocode failed",
            }
        adcode = geo.adcode or q.location_text
        display_city = self._display_name(q, geo)

        live_payload = self._weather_info(adcode, extensions="base") if q.include_live else {}
        forecast_payload = self._weather_info(adcode, extensions="all") if (q.include_forecast or q.days > 1 or q.include_travel_advice) else {}
        live = self._extract_live(live_payload)
        casts = self._extract_casts(forecast_payload, days=q.days)
        aqi = self._query_aqi(adcode) if q.include_aqi else {}
        warnings = self._query_warnings(adcode) if q.include_warning else []
        feels_like = self._estimate_feels_like(live)
        advice = self._travel_advice(q, live=live, casts=casts, aqi=aqi)

        ok = bool(live or casts)
        err = ""
        if not ok:
            err = live_payload.get("error") or forecast_payload.get("error") or "weather empty"
        return {
            "city": display_city,
            "ok": ok,
            "source": "amap_weather",
            "provider": "amap",
            "adcode": adcode,
            "location": geo.dict(),
            "gis": gis.dict(),
            "resolution_method": gis.method or "geocode",
            "query": q.dict(),
            "live": live,
            "feels_like": feels_like,
            "cast0": casts[0] if casts else {},
            "forecast": casts,
            "aqi": aqi,
            "warnings": warnings,
            "travel_advice": advice,
            "cache_backend": self.cache.backend,
            "error": err,
        }

    def _weather_info(self, adcode: str, *, extensions: str) -> Dict[str, Any]:
        key = f"amap-weather:{extensions}:{adcode}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            return cached
        if not self.api_key:
            return {"ok": False, "error": "AMAP_API_KEY is not configured"}
        try:
            params = {"key": self.api_key, "city": adcode, "extensions": extensions}
            resp = self._get("https://restapi.amap.com/v3/weather/weatherInfo", params=params, timeout=12)
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("status", "0")) != "1":
                payload = {"ok": False, "error": str(payload.get("info", "amap weather failed")), "raw": payload}
            self.cache.set_json(key, payload, ttl_seconds=600)
            return payload
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _extract_live(payload: Dict[str, Any]) -> Dict[str, Any]:
        lives = payload.get("lives") if isinstance(payload, dict) else None
        if isinstance(lives, list) and lives and isinstance(lives[0], dict):
            return dict(lives[0])
        return {}

    @staticmethod
    def _extract_casts(payload: Dict[str, Any], *, days: int) -> List[Dict[str, Any]]:
        forecasts = payload.get("forecasts") if isinstance(payload, dict) else None
        if not isinstance(forecasts, list) or not forecasts or not isinstance(forecasts[0], dict):
            return []
        casts = forecasts[0].get("casts")
        if not isinstance(casts, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in casts[: max(1, min(int(days or 1), 7))]:
            if isinstance(item, dict):
                out.append(dict(item))
        return out

    def _query_aqi(self, adcode: str) -> Dict[str, Any]:
        if not self.aqi_url or not self.api_key:
            return {"ok": False, "aqi": "", "quality": "暂无", "health_advice": "暂无空气质量数据，请以当地生态环境部门发布为准。"}
        key = f"amap-aqi:{adcode}"
        cached = self.cache.get_json(key)
        if isinstance(cached, dict):
            return cached
        try:
            resp = self._get(self.aqi_url, params={"key": self.api_key, "city": adcode}, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            result = self._normalize_aqi(payload)
            self.cache.set_json(key, result, ttl_seconds=600)
            return result
        except Exception as exc:
            return {"ok": False, "aqi": "", "quality": "暂无", "health_advice": f"空气质量接口暂不可用：{exc}"}

    @staticmethod
    def _normalize_aqi(payload: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = payload
        for key in ("data", "result", "live", "aqi"):
            val = row.get(key) if isinstance(row, dict) else None
            if isinstance(val, dict):
                row = val
        aqi = str(row.get("aqi", "") if isinstance(row, dict) else "").strip()
        quality = str(row.get("quality", "") or row.get("level", "") or "暂无").strip()
        advice = WeatherTool._aqi_advice(aqi, quality)
        return {"ok": bool(aqi or quality != "暂无"), "aqi": aqi, "quality": quality, "health_advice": advice, "raw": payload}

    def _query_warnings(self, adcode: str) -> List[Dict[str, Any]]:
        if not self.warning_url or not self.api_key:
            return []
        key = f"amap-warning:{adcode}"
        cached = self.cache.get_json(key)
        if isinstance(cached, list):
            return cached
        try:
            resp = self._get(self.warning_url, params={"key": self.api_key, "city": adcode}, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("warnings") or payload.get("data") or []
            result = rows if isinstance(rows, list) else []
            self.cache.set_json(key, result, ttl_seconds=600)
            return result
        except Exception:
            return []

    def _get(self, url: str, *, params: Dict[str, Any], timeout: int) -> requests.Response:
        if self.http_get is not None:
            return self.http_get(url, params=params, timeout=timeout, bypass_proxy=self.bypass_proxy)
        if not self.bypass_proxy:
            return requests.get(url, params=params, timeout=timeout)
        with requests.Session() as session:
            session.trust_env = False
            return session.get(url, params=params, timeout=timeout)

    @staticmethod
    def _display_name(q: ParsedWeatherQuery, geo: GeoCodeResult) -> str:
        if q.district:
            city = q.city or geo.city or geo.province
            return f"{city}{q.district}" if city and q.district not in city else q.district
        if geo.district:
            city = geo.city or geo.province
            return f"{city}{geo.district}" if city and geo.district not in city else geo.district
        return q.location_text or geo.formatted_address or geo.city or geo.province or geo.query

    @staticmethod
    def _estimate_feels_like(live: Dict[str, Any]) -> str:
        temp_raw = str(live.get("temperature", "") if isinstance(live, dict) else "").strip()
        if not temp_raw:
            return ""
        try:
            temp = float(temp_raw)
        except ValueError:
            return temp_raw
        humidity = 50.0
        try:
            hum_raw = str(live.get("humidity", "")).strip()
            if hum_raw:
                humidity = float(hum_raw)
        except ValueError:
            pass
        if temp >= 28:
            feels = temp + max(0.0, humidity - 55.0) * 0.05
        elif temp <= 8:
            feels = temp - 1.5
        else:
            feels = temp
        return str(int(round(feels)))

    @staticmethod
    def _aqi_advice(aqi: str, quality: str) -> str:
        try:
            val = int(float(aqi))
        except (TypeError, ValueError):
            val = 0
        q = quality or "暂无"
        if val and val <= 50 or q in {"优", "一级"}:
            return "空气质量较好，适合正常户外活动。"
        if val and val <= 100 or q in {"良", "二级"}:
            return "空气质量可接受，敏感人群可适当减少长时间剧烈运动。"
        if val and val <= 150 or "轻度" in q:
            return "轻度污染，老人、儿童和呼吸道敏感人群建议减少户外运动。"
        if val and val <= 200 or "中度" in q:
            return "污染偏高，建议减少户外停留，必要时佩戴防护口罩。"
        if val:
            return "污染较重，建议尽量减少外出并关闭门窗。"
        return "暂无空气质量数据，请以当地生态环境部门发布为准。"

    @staticmethod
    def _travel_advice(q: ParsedWeatherQuery, *, live: Dict[str, Any], casts: List[Dict[str, Any]], aqi: Dict[str, Any]) -> str:
        target_cast = casts[1] if q.travel_date == "tomorrow" and len(casts) > 1 else (casts[0] if casts else {})
        weather_text = " ".join(
            str(x)
            for x in [
                live.get("weather", ""),
                target_cast.get("dayweather", ""),
                target_cast.get("nightweather", ""),
            ]
            if x
        )
        temp = WeatherTool._pick_number(live.get("temperature")) or WeatherTool._pick_number(target_cast.get("daytemp"))
        wind_power = str(live.get("windpower", "") or target_cast.get("daypower", "") or "")
        parts: List[str] = []
        if any(k in weather_text for k in ("雨", "雪", "冰雹", "雷")):
            parts.append("有降水风险，建议携带雨具，穿防滑鞋，预留通勤时间。")
        else:
            parts.append("总体适合外出。")
        if temp is not None and temp >= 32:
            parts.append("气温较高，注意防晒补水，避免长时间暴晒。")
        elif temp is not None and temp <= 5:
            parts.append("气温较低，注意保暖防风。")
        if wind_power and ("5" in wind_power or "6" in wind_power or "7" in wind_power or "8" in wind_power):
            parts.append("风力偏大，骑行和高架桥通行需注意横风。")
        if aqi and aqi.get("quality") not in ("", "暂无", None):
            parts.append(str(aqi.get("health_advice", "")).strip())
        return "".join(p for p in parts if p)

    @staticmethod
    def _pick_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
