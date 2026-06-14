-- Weather GIS administrative boundary schema.
-- MySQL 8.0+ recommended. Geometry coordinates must be WGS84 lon/lat.

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Optional query sanity checks after import:
-- SELECT adcode,name,city,district
-- FROM weather_admin_boundary
-- WHERE ST_Contains(geom, ST_GeomFromText('POINT(39.9219 116.4431)', 4326));
--
-- SELECT b.adcode,b.name,b.city,b.district
-- FROM weather_admin_boundary b
-- JOIN weather_admin_alias a ON a.adcode=b.adcode
-- WHERE a.alias='西青区'
-- LIMIT 5;
