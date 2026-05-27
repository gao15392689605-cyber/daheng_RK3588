-- 烟草异物检测系统数据库初始化脚本
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    avatar_path TEXT DEFAULT '',
    create_time TIMESTAMP DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    detect_mode TEXT NOT NULL,
    file_path TEXT DEFAULT '',
    model_name TEXT NOT NULL,
    conf_threshold REAL NOT NULL,
    iou_threshold REAL NOT NULL,
    total_targets INTEGER DEFAULT 0,
    result_summary TEXT DEFAULT '{}',
    start_time TIMESTAMP DEFAULT (datetime('now','localtime')),
    end_time TIMESTAMP NULL,
    status TEXT DEFAULT 'success'
);

CREATE INDEX IF NOT EXISTS idx_username_time ON detection_logs(username, start_time);
