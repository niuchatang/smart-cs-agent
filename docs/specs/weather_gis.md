# Weather GIS Resolution

Weather queries resolve administrative areas in this order:

1. MySQL administrative lookup by `adcode/name/district/alias`.
2. MySQL GIS boundary table with `ST_Contains` for coordinate queries.
3. Local GeoJSON admin lookup or point-in-polygon.
3. Built-in development bboxes for the sample districts.
4. Geocode fallback when no GIS boundary source hits.

## MySQL Boundary Table

The production design uses two tables:

- `weather_admin_boundary`: one row per province/city/district boundary.
- `weather_admin_alias`: aliases like `西青区`, `西青`, `天津市西青区`, and adcode.

```sql
CREATE TABLE IF NOT EXISTS weather_admin_boundary (
  adcode VARCHAR(16) PRIMARY KEY,
  name VARCHAR(64) NOT NULL,
  province VARCHAR(64) NULL,
  city VARCHAR(64) NULL,
  district VARCHAR(64) NULL,
  level VARCHAR(32) NOT NULL,
  parent_adcode VARCHAR(16) NULL,
  center_lon DOUBLE NULL,
  center_lat DOUBLE NULL,
  geom MULTIPOLYGON NOT NULL SRID 4326,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  SPATIAL INDEX idx_weather_admin_boundary_geom (geom),
  INDEX idx_weather_admin_boundary_level (level),
  INDEX idx_weather_admin_boundary_city_district (city, district),
  INDEX idx_weather_admin_boundary_parent (parent_adcode)
);

CREATE TABLE IF NOT EXISTS weather_admin_alias (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  adcode VARCHAR(16) NOT NULL,
  alias VARCHAR(128) NOT NULL,
  alias_type VARCHAR(32) NOT NULL DEFAULT 'name',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_weather_admin_alias (alias, adcode),
  INDEX idx_weather_admin_alias_adcode (adcode),
  CONSTRAINT fk_weather_admin_alias_boundary
    FOREIGN KEY (adcode) REFERENCES weather_admin_boundary(adcode)
    ON DELETE CASCADE
);
```

Expected `level` values are `province`, `city`, and `district`.

Enable it with:

```env
GIS_ENABLE_MYSQL=true
GIS_BOUNDARY_TABLE=weather_admin_boundary
GIS_ALIAS_TABLE=weather_admin_alias
```

Apply schema:

```bash
mysql -u root -p smart_cs_agent < database/weather_gis_schema.sql
```

Import a nationwide WGS84 admin-boundary GeoJSON:

```bash
# 1) 下载（阿里云 DataV，约 500 个区县）
.venv/bin/python scripts/download_admin_boundaries.py \
  --output data/china_admin_boundaries.geojson

# 2) 导入 MySQL（脚本会自动读取 .env 中的 MYSQL_*）
.venv/bin/python scripts/import_admin_boundaries.py \
  --geojson data/china_admin_boundaries.geojson \
  --schema database/weather_gis_schema.sql \
  --truncate
```

> `/绝对路径/...` 是文档占位符，必须换成真实文件路径，例如 `data/china_admin_boundaries.geojson`。

After import:

- `西青区天气` can match `weather_admin_alias.alias='西青区'` and return `120111`.
- `116.4431,39.9219天气` uses `ST_Contains(geom, POINT(...))` and returns `110105`.
- `北京市海淀区天气` can match alias/direct table first and still use `adcode=110108`.

## GeoJSON Boundary File

Set:

```env
GIS_ADMIN_BOUNDARY_GEOJSON=/absolute/path/china_admin_boundaries.geojson
```

Each feature should have WGS84 geometry and properties like:

```json
{
  "adcode": "110105",
  "name": "朝阳区",
  "province": "北京市",
  "city": "北京市",
  "district": "朝阳区",
  "level": "district"
}
```
