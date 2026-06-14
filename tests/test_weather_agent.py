"""WeatherAgent and LocationParser behavior tests."""

from __future__ import annotations

from intent.weather_agent import WeatherAgent
from tools_infra.location_parser import LocationParser


def test_location_parser_city_weather() -> None:
    parsed = LocationParser().parse("北京天气怎么样")
    assert parsed.city == "北京市"
    assert parsed.location_text == "北京市"
    assert parsed.query_type == "current"


def test_location_parser_district_weather() -> None:
    parsed = LocationParser().parse("北京市朝阳区天气怎么样")
    assert parsed.province == "北京市"
    assert parsed.city == "北京市"
    assert parsed.district == "朝阳区"
    assert parsed.location_text == "北京市朝阳区"


def test_location_parser_future_days() -> None:
    parsed = LocationParser().parse("北京未来三天天气")
    assert parsed.days == 3
    assert parsed.include_forecast is True
    assert parsed.query_type == "forecast"


def test_location_parser_travel_advice() -> None:
    parsed = LocationParser().parse("明天去朝阳区适合出门吗")
    assert parsed.city == "北京市"
    assert parsed.district == "朝阳区"
    assert parsed.travel_date == "tomorrow"
    assert parsed.include_travel_advice is True


def test_location_parser_unknown_parent_district_keeps_district_name() -> None:
    parsed = LocationParser().parse("双流区天气")
    assert parsed.city == ""
    assert parsed.district == "双流区"
    assert parsed.location_text == "双流区"


def test_weather_agent_builds_district_tool_plan() -> None:
    plan = WeatherAgent(object()).try_plan("北京市海淀区天气怎么样", [])
    assert plan is not None
    assert plan["intent"] == "weather_query"
    assert plan["actions"][0]["tool"] == "query_weather"
    params = plan["actions"][0]["params"]
    assert params["cities"] == ["北京市海淀区"]
    assert params["weather_queries"][0]["district"] == "海淀区"


def test_weather_agent_air_quality_flag() -> None:
    plan = WeatherAgent(object()).try_plan("北京空气质量如何", [])
    assert plan is not None
    params = plan["actions"][0]["params"]
    assert params["include_aqi"] is True
    assert params["weather_queries"][0]["query_type"] == "air_quality"


def test_weather_agent_defers_route_weather() -> None:
    assert WeatherAgent(object()).try_plan("北京到上海天气怎么样", []) is None
