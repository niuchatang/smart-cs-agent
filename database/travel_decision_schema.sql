CREATE TABLE IF NOT EXISTS travel_decision_query_log (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT NULL,
  origin VARCHAR(128) NOT NULL,
  destination VARCHAR(128) NOT NULL,
  depart_time_text VARCHAR(128) NULL,
  travel_mode VARCHAR(32) NOT NULL,
  distance_km DECIMAL(10,2) NULL,
  duration_min INT NULL,
  risk_level VARCHAR(32) NULL,
  risk_score INT NULL,
  route_source VARCHAR(64) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_travel_decision_created (created_at),
  INDEX idx_travel_decision_od (origin, destination)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS travel_route_cache (
  cache_key VARCHAR(191) PRIMARY KEY,
  origin VARCHAR(128) NOT NULL,
  destination VARCHAR(128) NOT NULL,
  travel_mode VARCHAR(32) NOT NULL,
  payload JSON NOT NULL,
  expires_at DATETIME NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_travel_route_cache_expires (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
