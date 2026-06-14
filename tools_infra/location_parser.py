"""Natural-language location parsing for weather queries."""

from __future__ import annotations

import re
from typing import Dict, Optional

from pydantic import BaseModel, Field


_MUNICIPALITIES = {
    "北京": "北京市",
    "北京市": "北京市",
    "上海": "上海市",
    "上海市": "上海市",
    "天津": "天津市",
    "天津市": "天津市",
    "重庆": "重庆市",
    "重庆市": "重庆市",
}

_DISTRICT_DEFAULTS: Dict[str, Dict[str, str]] = {
    "朝阳区": {"province": "北京市", "city": "北京市"},
    "海淀区": {"province": "北京市", "city": "北京市"},
    "和平区": {"province": "天津市", "city": "天津市"},
    "河东区": {"province": "天津市", "city": "天津市"},
    "河西区": {"province": "天津市", "city": "天津市"},
    "南开区": {"province": "天津市", "city": "天津市"},
    "河北区": {"province": "天津市", "city": "天津市"},
    "红桥区": {"province": "天津市", "city": "天津市"},
    "东丽区": {"province": "天津市", "city": "天津市"},
    "西青区": {"province": "天津市", "city": "天津市"},
    "津南区": {"province": "天津市", "city": "天津市"},
    "北辰区": {"province": "天津市", "city": "天津市"},
    "武清区": {"province": "天津市", "city": "天津市"},
    "宝坻区": {"province": "天津市", "city": "天津市"},
    "滨海新区": {"province": "天津市", "city": "天津市"},
    "宁河区": {"province": "天津市", "city": "天津市"},
    "静海区": {"province": "天津市", "city": "天津市"},
    "蓟州区": {"province": "天津市", "city": "天津市"},
    "浦东新区": {"province": "上海市", "city": "上海市"},
    "余杭区": {"province": "浙江省", "city": "杭州市"},
    "南山区": {"province": "广东省", "city": "深圳市"},
}

_CITY_DEFAULTS: Dict[str, Dict[str, str]] = {
    "北京": {"province": "北京市", "city": "北京市"},
    "北京市": {"province": "北京市", "city": "北京市"},
    "上海": {"province": "上海市", "city": "上海市"},
    "上海市": {"province": "上海市", "city": "上海市"},
    "杭州": {"province": "浙江省", "city": "杭州市"},
    "杭州市": {"province": "浙江省", "city": "杭州市"},
    "深圳": {"province": "广东省", "city": "深圳市"},
    "深圳市": {"province": "广东省", "city": "深圳市"},
    "广州": {"province": "广东省", "city": "广州市"},
    "广州市": {"province": "广东省", "city": "广州市"},
}

_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


class ParsedWeatherQuery(BaseModel):
    raw_query: str = ""
    province: str = ""
    city: str = ""
    district: str = ""
    location_text: str = ""
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    days: int = Field(default=1, ge=1, le=7)
    include_live: bool = True
    include_forecast: bool = False
    include_aqi: bool = False
    include_warning: bool = False
    include_travel_advice: bool = False
    travel_date: str = "today"
    query_type: str = "current"
    is_weather_query: bool = False

    def to_tool_params(self) -> Dict[str, object]:
        return {
            "cities": [self.location_text] if self.location_text else [],
            "weather_queries": [self.dict()],
            "days": self.days,
            "include_aqi": self.include_aqi,
            "include_forecast": self.include_forecast,
            "include_warning": self.include_warning,
            "include_travel_advice": self.include_travel_advice,
            "travel_date": self.travel_date,
            "longitude": self.longitude,
            "latitude": self.latitude,
        }


class LocationParser:
    """Extract province/city/district and weather query options from Chinese text."""

    WEATHER_KEYWORDS = (
        "天气",
        "气温",
        "温度",
        "湿度",
        "风力",
        "降雨",
        "下雨",
        "下雪",
        "空气质量",
        "AQI",
        "aqi",
        "雾霾",
        "适合出门",
        "出行建议",
        "带伞",
        "热不热",
        "冷不冷",
        "预警",
    )

    def parse(self, message: str) -> ParsedWeatherQuery:
        raw = (message or "").strip()
        days = self._extract_days(raw)
        include_aqi = bool(re.search(r"(空气质量|AQI|aqi|雾霾|PM2\.?5|污染)", raw))
        include_warning = "预警" in raw or any(k in raw for k in ("台风", "暴雨", "大风", "冰雹"))
        include_travel = bool(re.search(r"(适合出门|出门|出行|带伞|雨具|去.+吗)", raw))
        travel_date = "tomorrow" if "明天" in raw else ("after_tomorrow" if "后天" in raw else "today")
        include_forecast = days > 1 or "未来" in raw or "预报" in raw or travel_date != "today"
        lon, lat = self._extract_coordinates(raw)

        loc = self._extract_location_text(raw)
        parts = {"location_text": loc} if lon is not None and lat is not None else self._split_location(loc)
        query_type = self._query_type(include_aqi, include_travel, include_forecast)
        return ParsedWeatherQuery(
            raw_query=raw,
            province=parts.get("province", ""),
            city=parts.get("city", ""),
            district=parts.get("district", ""),
            location_text=parts.get("location_text", loc),
            longitude=lon,
            latitude=lat,
            days=days,
            include_live=True,
            include_forecast=include_forecast,
            include_aqi=include_aqi,
            include_warning=include_warning,
            include_travel_advice=include_travel,
            travel_date=travel_date,
            query_type=query_type,
            is_weather_query=self.is_weather_query(raw),
        )

    @classmethod
    def is_weather_query(cls, message: str) -> bool:
        return any(k in (message or "") for k in cls.WEATHER_KEYWORDS)

    @staticmethod
    def _extract_coordinates(text: str) -> tuple[Optional[float], Optional[float]]:
        raw = text or ""
        labeled = re.search(
            r"(?:经度|lon|lng|longitude)\s*[:：]?\s*(-?\d+(?:\.\d+)?)"
            r".{0,12}?"
            r"(?:纬度|lat|latitude)\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
            raw,
            flags=re.IGNORECASE,
        )
        if labeled:
            return float(labeled.group(1)), float(labeled.group(2))

        labeled_rev = re.search(
            r"(?:纬度|lat|latitude)\s*[:：]?\s*(-?\d+(?:\.\d+)?)"
            r".{0,12}?"
            r"(?:经度|lon|lng|longitude)\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
            raw,
            flags=re.IGNORECASE,
        )
        if labeled_rev:
            return float(labeled_rev.group(2)), float(labeled_rev.group(1))

        pair = re.search(r"(-?\d{2,3}(?:\.\d+)?)\s*[,，\s]\s*(-?\d{2,3}(?:\.\d+)?)", raw)
        if not pair:
            return None, None
        a = float(pair.group(1))
        b = float(pair.group(2))
        if 70 <= a <= 140 and 15 <= b <= 55:
            return a, b
        if 70 <= b <= 140 and 15 <= a <= 55:
            return b, a
        return None, None

    @staticmethod
    def _query_type(include_aqi: bool, include_travel: bool, include_forecast: bool) -> str:
        if include_aqi:
            return "air_quality"
        if include_travel:
            return "travel"
        if include_forecast:
            return "forecast"
        return "current"

    @classmethod
    def _extract_days(cls, text: str) -> int:
        if "后天" in text:
            return 3
        if "明天" in text:
            return 2
        patterns = (
            r"未来\s*([一二两三四五六七八九十\d]+)\s*天",
            r"近\s*([一二两三四五六七八九十\d]+)\s*天",
            r"([一二两三四五六七八九十\d]+)\s*天(?:天气|预报)",
        )
        for pat in patterns:
            m = re.search(pat, text)
            if not m:
                continue
            n = cls._parse_cn_number(m.group(1))
            if n:
                return min(max(n, 1), 7)
        return 1

    @staticmethod
    def _parse_cn_number(token: str) -> Optional[int]:
        t = (token or "").strip()
        if not t:
            return None
        if t.isdigit():
            return int(t)
        if t in _CN_NUM:
            return _CN_NUM[t]
        if t.startswith("十") and len(t) == 2:
            return 10 + _CN_NUM.get(t[1], 0)
        if "十" in t:
            left, _, right = t.partition("十")
            return _CN_NUM.get(left, 1) * 10 + (_CN_NUM.get(right, 0) if right else 0)
        return None

    def _extract_location_text(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        lon, lat = self._extract_coordinates(s)
        if lon is not None and lat is not None:
            return f"{lon:.6f},{lat:.6f}"

        go = re.search(r"(?:去|到)\s*([\u4e00-\u9fff]{2,18}?)(?:适合|天气|出门|玩|吗|呢|如何|怎么样|$)", s)
        if go:
            return self._clean_location_candidate(go.group(1))

        markers = ("空气质量", "AQI", "aqi", "天气", "气温", "温度", "湿度", "风力", "降雨", "下雨", "下雪", "预警")
        for marker in markers:
            idx = s.find(marker)
            if idx > 0:
                candidate = s[:idx]
                candidate = self._clean_location_candidate(candidate)
                if candidate:
                    return candidate

        return self._clean_location_candidate(s)

    @staticmethod
    def _clean_location_candidate(value: str) -> str:
        t = re.sub(r"\s+", "", (value or "").strip())
        t = re.sub(r"^(帮我|请|请问|查一下|查下|查询|看看|看下|我想知道|想知道)", "", t)
        t = re.sub(r"(今天|现在|当前|实时|明天|后天|未来[一二两三四五六七八九十\d]+天|近[一二两三四五六七八九十\d]+天)", "", t)
        t = re.sub(r"(怎么样|如何|怎样|好吗|好么|吗|呢|呀|啊|吧|？|\?|。|，|,|！|!)", "", t)
        t = re.sub(r"(的)?(天气|气温|温度|空气质量|AQI|aqi|预报|出行建议)$", "", t)
        t = t.strip()
        return t

    @staticmethod
    def _ensure_city_suffix(name: str) -> str:
        n = (name or "").strip()
        if not n:
            return ""
        if n in _MUNICIPALITIES:
            return _MUNICIPALITIES[n]
        if n.endswith(("市", "州", "盟", "地区")):
            return n
        return f"{n}市"

    def _split_location(self, location: str) -> Dict[str, str]:
        loc = self._clean_location_candidate(location)
        if not loc:
            return {"location_text": ""}

        if loc in _CITY_DEFAULTS:
            city_meta = _CITY_DEFAULTS[loc]
            return {
                "province": city_meta["province"],
                "city": city_meta["city"],
                "district": "",
                "location_text": city_meta["city"],
            }

        for prefix, full_city in sorted(_MUNICIPALITIES.items(), key=lambda item: len(item[0]), reverse=True):
            if loc.startswith(prefix):
                rest = loc[len(prefix) :].strip()
                district = rest if rest.endswith(("区", "县", "新区", "旗")) else ""
                return {
                    "province": full_city,
                    "city": full_city,
                    "district": district,
                    "location_text": f"{full_city}{district}" if district else full_city,
                }

        for district, meta in _DISTRICT_DEFAULTS.items():
            if loc == district or loc.endswith(district):
                return {
                    "province": meta.get("province", ""),
                    "city": meta.get("city", ""),
                    "district": district,
                    "location_text": f"{meta.get('city', '')}{district}",
                }

        m_city_district = re.match(r"^([\u4e00-\u9fff]{2,10}?市)([\u4e00-\u9fff]{2,10}(?:新区|区|县|旗))$", loc)
        if m_city_district:
            city = self._ensure_city_suffix(m_city_district.group(1))
            district = m_city_district.group(2)
            return {"province": "", "city": city, "district": district, "location_text": f"{city}{district}"}

        m_plain = re.match(r"^([\u4e00-\u9fff]{2,4})([\u4e00-\u9fff]{2,10}(?:新区|区|县|旗))$", loc)
        if m_plain:
            city = self._ensure_city_suffix(m_plain.group(1))
            district = m_plain.group(2)
            return {"province": "", "city": city, "district": district, "location_text": f"{city}{district}"}

        if re.match(r"^[\u4e00-\u9fff]{2,12}(?:新区|区|县|旗)$", loc):
            return {
                "province": "",
                "city": "",
                "district": loc,
                "location_text": loc,
            }

        city = self._ensure_city_suffix(loc)
        meta = _CITY_DEFAULTS.get(city) or _CITY_DEFAULTS.get(loc) or {}
        return {
            "province": meta.get("province", ""),
            "city": meta.get("city", city),
            "district": "",
            "location_text": meta.get("city", city),
        }
