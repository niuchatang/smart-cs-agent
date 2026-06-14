"""Weather tool tests with fake AMap responses."""

from __future__ import annotations

from typing import Any, Dict

from tools_infra.location_parser import LocationParser
from tools_infra.weather_cache import WeatherCache
from tools_infra.weather_tools import GeoCodeTool, WeatherTool


class _Resp:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def _fake_http_get(url: str, *, params: Dict[str, Any], timeout: int, bypass_proxy: bool) -> _Resp:
    assert timeout > 0
    assert bypass_proxy in {True, False}
    if "weatherInfo" in url and params.get("extensions") == "base":
        return _Resp(
            {
                "status": "1",
                "lives": [
                    {
                        "province": "北京市",
                        "city": "朝阳区",
                        "adcode": "110105",
                        "weather": "多云",
                        "temperature": "28",
                        "winddirection": "东南",
                        "windpower": "3",
                        "humidity": "55",
                    }
                ],
            }
        )
    if "weatherInfo" in url and params.get("extensions") == "all":
        return _Resp(
            {
                "status": "1",
                "forecasts": [
                    {
                        "city": "朝阳区",
                        "adcode": "110105",
                        "casts": [
                            {"date": "2026-06-14", "dayweather": "多云", "nightweather": "晴", "daytemp": "30", "nighttemp": "22"},
                            {"date": "2026-06-15", "dayweather": "小雨", "nightweather": "小雨", "daytemp": "27", "nighttemp": "21"},
                            {"date": "2026-06-16", "dayweather": "晴", "nightweather": "晴", "daytemp": "31", "nighttemp": "23"},
                        ],
                    }
                ],
            }
        )
    return _Resp({"status": "0", "info": "unexpected"})


def test_geocode_builtin_district() -> None:
    cache = WeatherCache(ttl_seconds=600, namespace="test-weather-geocode")
    geo = GeoCodeTool(api_key="", cache=cache).geocode("北京市朝阳区")
    assert geo.ok is True
    assert geo.adcode == "110105"
    assert geo.district == "朝阳区"


def test_weather_tool_amap_live_and_forecast() -> None:
    cache = WeatherCache(ttl_seconds=600, namespace="test-weather-tool")
    geocode = GeoCodeTool(api_key="fake", cache=cache, http_get=_fake_http_get)
    tool = WeatherTool(api_key="fake", cache=cache, geocode_tool=geocode, http_get=_fake_http_get)
    parsed = LocationParser().parse("北京市朝阳区未来三天天气")
    result = tool.query(parsed)
    assert result["ok"] is True
    assert result["adcode"] == "110105"
    assert result["live"]["weather"] == "多云"
    assert result["feels_like"]
    assert len(result["forecast"]) == 3
    assert "适合外出" in result["travel_advice"] or "降水风险" in result["travel_advice"]
    assert result["resolution_method"].startswith("gis_")


def test_weather_tool_coordinate_query_uses_gis_adcode() -> None:
    cache = WeatherCache(ttl_seconds=600, namespace="test-weather-tool-coordinate")
    geocode = GeoCodeTool(api_key="fake", cache=cache, http_get=_fake_http_get)
    tool = WeatherTool(api_key="fake", cache=cache, geocode_tool=geocode, http_get=_fake_http_get)
    parsed = LocationParser().parse("116.4431,39.9219 天气")
    result = tool.query(parsed)
    assert result["ok"] is True
    assert result["adcode"] == "110105"
    assert result["gis"]["method"] == "gis_builtin_bbox"
