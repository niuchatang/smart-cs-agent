"""Tools for Travel Decision Agent.

The module keeps travel-risk parsing, route GIS extraction, risk scoring, and
response formatting separate from the FastAPI service and from intent routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .gis_location_tool import AdminAreaResult, GISLocationTool
from .location_parser import LocationParser


class TravelDecisionQuery(BaseModel):
    raw_query: str = ""
    origin: str = ""
    destination: str = ""
    depart_time_text: str = ""
    travel_mode: str = "driving"
    origin_parts: Dict[str, Any] = Field(default_factory=dict)
    destination_parts: Dict[str, Any] = Field(default_factory=dict)


class RouteArea(BaseModel):
    adcode: str = ""
    name: str = ""
    province: str = ""
    city: str = ""
    district: str = ""
    level: str = ""
    source: str = ""
    method: str = ""

    @property
    def label(self) -> str:
        if self.city and self.district:
            return f"{self.city}{self.district}"
        if self.district:
            return self.district
        if self.city:
            return self.city
        return self.name or self.province or self.adcode


class RiskAssessment(BaseModel):
    risk_score: int = 0
    risk_level: str = "Low"
    risk_level_cn: str = "低"
    main_risks: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class TravelQueryParser:
    """Parse origin, destination, time text, and travel mode from Chinese text."""

    _TIME_PATTERN = re.compile(
        r"(今天|明天|后天|周[一二三四五六日天]|星期[一二三四五六日天])?"
        r"(凌晨|早上|上午|中午|下午|傍晚|晚上|夜间)?"
        r"(\d{1,2}\s*[点:：]\s*\d{0,2}分?)?"
    )

    def __init__(self, location_parser: LocationParser | None = None) -> None:
        self.location_parser = location_parser or LocationParser()

    def parse(self, message: str) -> Optional[TravelDecisionQuery]:
        raw = (message or "").strip()
        if not raw:
            return None
        od = self._extract_od(raw)
        if od is None:
            return None
        origin_raw, dest_raw = od
        origin = self.location_parser.parse(f"{origin_raw}天气")
        dest = self.location_parser.parse(f"{dest_raw}天气")
        return TravelDecisionQuery(
            raw_query=raw,
            origin=origin.location_text or origin_raw,
            destination=dest.location_text or dest_raw,
            depart_time_text=self._extract_depart_time(raw),
            travel_mode=self._extract_mode(raw),
            origin_parts=origin.model_dump(),
            destination_parts=dest.model_dump(),
        )

    @staticmethod
    def is_travel_decision_query(message: str) -> bool:
        text = message or ""
        has_od = ("从" in text and ("到" in text or "去" in text)) or re.search(r".+到.+", text) is not None
        decisionish = any(
            k in text
            for k in (
                "适合几点",
                "几点出发",
                "什么时候出发",
                "出发时间",
                "适合出发",
                "出行建议",
                "风险",
                "好走",
                "能不能去",
                "适合去",
            )
        )
        modeish = any(k in text for k in ("开车", "自驾", "驾车", "出发", "去", "到"))
        return bool(has_od and (decisionish or modeish))

    @staticmethod
    def _extract_od(text: str) -> Optional[tuple[str, str]]:
        patterns = (
            r"从(.+?)(?:开车|自驾|驾车|坐车|乘车|步行|骑行)?(?:去|到)(.+?)(?:，|,|。|；|;|适合|几点|什么时候|怎么走|$)",
            r"(.+?)(?:开车|自驾|驾车)?到(.+?)(?:，|,|。|；|;|适合|几点|什么时候|怎么走|$)",
        )
        for pat in patterns:
            m = re.search(pat, text)
            if not m:
                continue
            origin = TravelQueryParser._clean_place_token(m.group(1))
            dest = TravelQueryParser._clean_place_token(m.group(2))
            if origin and dest and origin != dest:
                return origin, dest
        return None

    @staticmethod
    def _clean_place_token(value: str) -> str:
        text = re.sub(r"\s+", "", (value or "").strip())
        text = re.sub(r"^(今天|明天|后天|上午|下午|早上|晚上|中午|夜间)+", "", text)
        text = re.sub(r"(路线|天气|路况|风险|出行建议)$", "", text)
        return text.strip("，,。；; ")

    @classmethod
    def _extract_depart_time(cls, text: str) -> str:
        candidates: List[str] = []
        for m in cls._TIME_PATTERN.finditer(text):
            token = "".join(x or "" for x in m.groups()).strip()
            if token:
                candidates.append(token)
        for token in candidates:
            if any(k in token for k in ("今天", "明天", "后天", "上午", "下午", "早上", "晚上", "点", "周", "星期")):
                return token
        return ""

    @staticmethod
    def _extract_mode(text: str) -> str:
        if any(k in text for k in ("公交", "地铁", "公共交通")):
            return "transit"
        if "步行" in text:
            return "walking"
        if "骑行" in text or "自行车" in text:
            return "cycling"
        return "driving"


class RouteGISService:
    """Resolve route points and labels to administrative regions."""

    def __init__(self, gis_tool: GISLocationTool, geocode_tool: Any | None = None) -> None:
        self.gis_tool = gis_tool
        self.geocode_tool = geocode_tool

    def areas_from_route(
        self,
        route_result: Dict[str, Any],
        *,
        origin: str,
        destination: str,
        max_points: int = 8,
    ) -> List[Dict[str, Any]]:
        areas: List[RouteArea] = []
        for name in (origin, destination):
            hit = self.gis_tool.resolve_admin_name(name)
            if not hit.ok and self.geocode_tool is not None:
                parsed = LocationParser().parse(f"{name}天气")
                hit = self.gis_tool.resolve_query(parsed, geocode_tool=self.geocode_tool)
            self._append_hit(areas, hit)

        pts = route_result.get("route_points")
        if isinstance(pts, list) and len(pts) >= 2:
            for p in self._sample_points(pts, max_points=max_points):
                hit = self.gis_tool.resolve_point(p["lon"], p["lat"], preferred_level="district")
                self._append_hit(areas, hit)

        city_labels = route_result.get("cities_along_route")
        if isinstance(city_labels, list):
            for label in city_labels[:max_points]:
                hit = self.gis_tool.resolve_admin_name(str(label).strip())
                self._append_hit(areas, hit)

        return [a.model_dump() for a in areas]

    @staticmethod
    def _sample_points(points: List[Any], *, max_points: int) -> List[Dict[str, float]]:
        if not points:
            return []
        n = len(points)
        if n == 1:
            indices = [0]
        else:
            count = max(2, min(max_points, n))
            indices = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
        out: List[Dict[str, float]] = []
        for idx in indices:
            raw = points[int(idx)]
            if not isinstance(raw, list) or len(raw) < 2:
                continue
            try:
                lat = float(raw[0])
                lon = float(raw[1])
            except (TypeError, ValueError):
                continue
            out.append({"lat": lat, "lon": lon})
        return out

    @staticmethod
    def _append_hit(areas: List[RouteArea], hit: AdminAreaResult) -> None:
        if not hit.ok:
            return
        area = RouteArea(
            adcode=hit.adcode,
            name=hit.name,
            province=hit.province,
            city=hit.city,
            district=hit.district,
            level=hit.level,
            source=hit.source,
            method=hit.method,
        )
        key = area.adcode or area.label
        if not key:
            return
        existing = {a.adcode or a.label for a in areas}
        if key not in existing:
            areas.append(area)


class RiskScorer:
    """Rule-based route risk scorer."""

    @classmethod
    def score(
        cls,
        *,
        route_result: Dict[str, Any],
        weather_blocks: List[Dict[str, Any]],
        highway_results: List[Dict[str, Any]],
    ) -> RiskAssessment:
        score = 0
        risks: List[str] = []

        for block in weather_blocks:
            if not isinstance(block, dict) or not block.get("ok"):
                continue
            city = str(block.get("city") or "沿途").strip()
            text = cls._weather_text(block)
            if "暴雨" in text:
                score += 50
                risks.append(f"{city}有暴雨风险")
            elif "大雨" in text:
                score += 30
                risks.append(f"{city}有大雨风险")
            if any(k in text for k in ("大风", "强风", "阵风")):
                score += 20
                risks.append(f"{city}风力偏大")
            if any(k in text for k in ("大雾", "浓雾", "雾")):
                score += 40
                risks.append(f"{city}能见度可能受雾影响")
            aqi = cls._aqi_value(block.get("aqi"))
            if aqi is not None and aqi > 200:
                score += 20
                risks.append(f"{city}AQI>{aqi:.0f}，空气质量风险较高")
            visibility = cls._visibility_m(block)
            if visibility is not None and visibility < 500:
                score += 40
                risks.append(f"{city}能见度低于500米")

        for hw in highway_results:
            if not isinstance(hw, dict) or not hw.get("success"):
                continue
            congestion = str(hw.get("congestion_level") or hw.get("status") or "")
            if any(k in congestion for k in ("拥堵", "缓行", "重度", "中度")):
                score += 30
                target = str(hw.get("target") or hw.get("name") or "高速").strip()
                risks.append(f"{target}存在{congestion}")

        if route_result.get("warning"):
            risks.append(str(route_result.get("warning")))

        level, level_cn = cls._level(score)
        return RiskAssessment(
            risk_score=score,
            risk_level=level,
            risk_level_cn=level_cn,
            main_risks=risks[:6],
            suggestions=cls._suggestions(level, risks),
        )

    @staticmethod
    def _weather_text(block: Dict[str, Any]) -> str:
        live = block.get("live") if isinstance(block.get("live"), dict) else {}
        casts = block.get("forecast") if isinstance(block.get("forecast"), list) else []
        parts = [str(live.get("weather") or "")]
        for cast in casts[:2]:
            if isinstance(cast, dict):
                parts.append(str(cast.get("dayweather") or ""))
                parts.append(str(cast.get("nightweather") or ""))
        return " ".join(parts)

    @staticmethod
    def _aqi_value(aqi: Any) -> Optional[float]:
        if not isinstance(aqi, dict):
            return None
        raw = aqi.get("aqi")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _visibility_m(block: Dict[str, Any]) -> Optional[float]:
        live = block.get("live") if isinstance(block.get("live"), dict) else {}
        raw = live.get("visibility") or live.get("vis")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _level(score: int) -> tuple[str, str]:
        if score >= 100:
            return "Critical", "严重"
        if score >= 70:
            return "High", "较高"
        if score >= 30:
            return "Medium", "中等"
        return "Low", "低"

    @staticmethod
    def _suggestions(level: str, risks: List[str]) -> List[str]:
        out: List[str] = []
        if level in {"High", "Critical"}:
            out.append("建议推迟或调整路线，避开高风险时段和路段。")
        elif level == "Medium":
            out.append("建议提前出发，预留缓行和天气影响时间。")
        else:
            out.append("总体适合出行，出发前再确认实时路况。")
        if any("雨" in r for r in risks):
            out.append("建议携带雨具，雨天降低车速并保持车距。")
        if any("雾" in r or "能见度" in r for r in risks):
            out.append("雾天开启雾灯，避免急刹和频繁变道。")
        if any("AQI" in r for r in risks):
            out.append("空气质量较差时，敏感人群减少长时间户外停留。")
        if not any("09:00" in x for x in out):
            out.append("若行程允许，优先选择上午较早时段出发。")
        return out[:5]


class TravelDecisionFormatter:
    @staticmethod
    def format(result: Dict[str, Any]) -> str:
        if not result.get("success"):
            return f"暂时无法完成出行决策分析：{result.get('error', '未知错误')}"

        route = result.get("route") if isinstance(result.get("route"), dict) else {}
        query = result.get("query") if isinstance(result.get("query"), dict) else {}
        risk = result.get("risk") if isinstance(result.get("risk"), dict) else {}
        areas = result.get("route_areas") if isinstance(result.get("route_areas"), list) else []
        weather = result.get("weather") if isinstance(result.get("weather"), list) else []

        lines: List[str] = ["【出行分析】"]
        lines.append(f"起点：{query.get('origin') or route.get('origin') or '未知'}")
        lines.append(f"终点：{query.get('destination') or route.get('destination') or '未知'}")
        if query.get("depart_time_text"):
            lines.append(f"出发时间：{query.get('depart_time_text')}")
        lines.append(f"交通方式：{TravelDecisionFormatter._mode_cn(str(query.get('travel_mode') or 'driving'))}")
        lines.append(f"预计距离：{route.get('distance_km', '?')}公里")
        lines.append(f"预计时间：{TravelDecisionFormatter._duration_cn(route.get('duration_min'))}")
        highways = route.get("highways") if isinstance(route.get("highways"), list) else []
        if highways:
            lines.append(f"高速信息：{'、'.join(str(x) for x in highways[:6])}")
        service_areas = route.get("service_areas") if isinstance(route.get("service_areas"), list) else []
        if service_areas:
            names = [str(x.get("name") or x.get("title") or "").strip() for x in service_areas if isinstance(x, dict)]
            names = [x for x in names if x]
            if names:
                lines.append(f"服务区信息：{'、'.join(names[:5])}")
        if areas:
            labels = [TravelDecisionFormatter._area_label(x) for x in areas if isinstance(x, dict)]
            labels = [x for x in labels if x]
            lines.append(f"途经区域：{'、'.join(labels[:12])}")

        lines.append("")
        lines.append("【天气情况】")
        if weather:
            for block in weather[:8]:
                if not isinstance(block, dict) or not block.get("ok"):
                    continue
                name = str(block.get("city") or "沿途").strip()
                live = block.get("live") if isinstance(block.get("live"), dict) else {}
                desc = str(live.get("weather") or "").strip()
                temp = str(live.get("temperature") or "").strip()
                suffix = f"，{temp}℃" if temp else ""
                lines.append(f"{name}：{desc or '暂无'}{suffix}")
        else:
            lines.append("沿途天气暂不可用。")

        lines.append("")
        lines.append("【风险评估】")
        level = risk.get("risk_level") or "Low"
        level_cn = risk.get("risk_level_cn") or "低"
        lines.append(f"综合风险：{level}（{level_cn}）")
        main_risks = risk.get("main_risks") if isinstance(risk.get("main_risks"), list) else []
        if main_risks:
            lines.append(f"主要风险：{'；'.join(str(x) for x in main_risks[:4])}")
        else:
            lines.append("主要风险：暂无明显高风险因素")

        lines.append("")
        lines.append("【出发建议】")
        suggestions = risk.get("suggestions") if isinstance(risk.get("suggestions"), list) else []
        if suggestions:
            lines.append(f"推荐：{suggestions[0]}")
            for item in suggestions[1:]:
                lines.append(str(item))
        else:
            lines.append("推荐：出发前再次确认实时天气、导航路况和预警信息。")
        return "\n".join(lines).strip()

    @staticmethod
    def _area_label(area: Dict[str, Any]) -> str:
        city = str(area.get("city") or "").strip()
        district = str(area.get("district") or "").strip()
        if city and district:
            return f"{city}{district}"
        return district or city or str(area.get("name") or "").strip()

    @staticmethod
    def _duration_cn(value: Any) -> str:
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            return "未知"
        h, m = divmod(max(minutes, 0), 60)
        if h:
            return f"{h}小时{m}分钟"
        return f"{m}分钟"

    @staticmethod
    def _mode_cn(mode: str) -> str:
        return {"driving": "自驾", "transit": "公共交通", "walking": "步行", "cycling": "骑行"}.get(mode, mode)
