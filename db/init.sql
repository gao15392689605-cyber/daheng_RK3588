-- 烟草异物检测系统数据库初始化脚本
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    avatar_path TEXT DEFAULT '',
    role TEXT DEFAULT 'worker',            -- worker / technician / admin
    must_change_pwd INTEGER DEFAULT 0,     -- 1=首次登录须改密
    active INTEGER DEFAULT 1,              -- 1=启用 0=停用
    full_name TEXT DEFAULT '',
    emp_id TEXT DEFAULT '',                -- 工号(人工填写, 非自增 id)
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

-- 审计日志:记录"谁、何时、对系统做了什么关键操作"(登录/改参数/换模型/账号操作)
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TIMESTAMP DEFAULT (datetime('now','localtime')),
    username TEXT DEFAULT '',
    role TEXT DEFAULT '',
    action TEXT NOT NULL,          -- 操作类型: login/logout/param_change/model_switch/user_create/...
    detail TEXT DEFAULT ''         -- 详情(如 "conf 0.50→0.40")
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(time);

-- 报警记录:严重异物报警 + 处置闭环(谁、何时、什么异物、是否分流、处置结果)
CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TIMESTAMP DEFAULT (datetime('now','localtime')),
    username TEXT DEFAULT '',
    class_eng TEXT DEFAULT '',
    class_cn TEXT DEFAULT '',
    severity TEXT DEFAULT 'severe',   -- severe / normal
    diverted INTEGER DEFAULT 0,       -- 1=已触发下游分流
    handled INTEGER DEFAULT 0,        -- 1=工人已确认处置
    action TEXT DEFAULT '',           -- 处置结果: 上游正常 / 发现损坏 / 已拣出 ...
    image_path TEXT DEFAULT '',       -- 证据帧路径
    batch_id TEXT DEFAULT ''          -- 预留: 批次追溯(Step5)
);
CREATE INDEX IF NOT EXISTS idx_alarm_time ON alarms(time);

-- 批次:一段有标识的检测会话。开批次时快照"用的配置"(模型+阈值),期间报警挂 batch_id,
-- 结束时汇总,供质量追溯(出问题按批次号倒查:用的哪套配置、检出哪些异物、谁当班)
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,            -- 批次号/工单号(人工录入)
    variety TEXT DEFAULT '',           -- 品种
    line TEXT DEFAULT '',              -- 产线
    operator TEXT DEFAULT '',          -- 当班人
    model_name TEXT DEFAULT '',        -- 配置快照: 模型
    conf REAL DEFAULT 0,               -- 配置快照: conf
    iou REAL DEFAULT 0,                -- 配置快照: iou
    start_time TIMESTAMP DEFAULT (datetime('now','localtime')),
    end_time TIMESTAMP NULL,
    status TEXT DEFAULT 'running',     -- running / done
    alarm_count INTEGER DEFAULT 0,
    total_targets INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_batch_id ON batches(batch_id);
