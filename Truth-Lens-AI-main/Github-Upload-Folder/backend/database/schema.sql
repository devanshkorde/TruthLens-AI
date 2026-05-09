-- TruthLens AI — Database Schema

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_type TEXT NOT NULL,           -- 'text' | 'image' | 'url'
    input_summary TEXT,                  -- first 200 chars of text or filename/url
    verdict TEXT NOT NULL,               -- 'FAKE' | 'REAL' | 'SUSPICIOUS'
    confidence REAL NOT NULL,            -- 0.0 – 1.0
    credibility_score INTEGER,           -- 0 – 100
    signals TEXT,                        -- JSON array of signal strings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    status_code INTEGER,
    processing_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
