"""GIS location resolver tests."""

from __future__ import annotations

from tools_infra.gis_location_tool import GISLocationTool
from tools_infra.location_parser import LocationParser
from tools_infra.weather_cache import WeatherCache
from tools_infra.weather_tools import GeoCodeTool


def test_parse_coordinate_weather_query() -> None:
    parsed = LocationParser().parse("经度116.4431 纬度39.9219天气")
    assert parsed.longitude == 116.4431
    assert parsed.latitude == 39.9219
    assert parsed.location_text == "116.443100,39.921900"


def test_gis_builtin_point_resolves_district() -> None:
    tool = GISLocationTool(cache=WeatherCache(ttl_seconds=600, namespace="test-gis-point"))
    hit = tool.resolve_point(116.4431, 39.9219)
    assert hit.ok is True
    assert hit.method == "gis_builtin_bbox"
    assert hit.adcode == "110105"
    assert hit.district == "朝阳区"


def test_gis_query_uses_geocode_point_then_polygon() -> None:
    cache = WeatherCache(ttl_seconds=600, namespace="test-gis-query")
    parsed = LocationParser().parse("北京市朝阳区天气")
    hit = GISLocationTool(cache=cache).resolve_query(parsed, geocode_tool=GeoCodeTool(api_key="", cache=cache))
    assert hit.ok is True
    assert hit.method == "gis_builtin_admin_lookup"
    assert hit.adcode == "110105"


def test_tianjin_district_text_resolves_by_gis_admin_lookup() -> None:
    cache = WeatherCache(ttl_seconds=600, namespace="test-gis-tianjin")
    parser = LocationParser()
    geo = GeoCodeTool(api_key="", cache=cache)
    gis = GISLocationTool(cache=cache)

    xiqing = parser.parse("西青区天气")
    assert xiqing.location_text == "天津市西青区"
    hit1 = gis.resolve_query(xiqing, geocode_tool=geo)
    assert hit1.ok is True
    assert hit1.adcode == "120111"
    assert hit1.method == "gis_builtin_admin_lookup"

    jinnan = parser.parse("津南区天气")
    assert jinnan.location_text == "天津市津南区"
    hit2 = gis.resolve_query(jinnan, geocode_tool=geo)
    assert hit2.ok is True
    assert hit2.adcode == "120112"
    assert hit2.method == "gis_builtin_admin_lookup"
