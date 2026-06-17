#!/usr/bin/env bash
# 烟草异物检测系统 — 一键重置(清缓存 + 清数据库, 从头开始)
# 用法:  bash 重置.sh
# 作用:  删数据库/产线配置/Python缓存/报警证据帧/工人拍摄图;
#        下次启动会重建空库并预置默认管理员 admin / admin123。

set -u

# 脚本所在目录(= tobacco_detection_system), 不管在哪运行都对
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$HERE")"
CAPTURE_DIR="$PARENT/异物图"   # 工人拍摄存图目录(与 config.CAPTURE_DIR 一致)

echo "============================================"
echo " 烟草异物检测系统 — 重置"
echo "============================================"
echo "当前系统时间: $(date '+%Y-%m-%d %H:%M:%S')  (时区: $(cat /etc/timezone 2>/dev/null || echo 未知))"
echo
echo "⚠ RK3588 无 RTC 电池, 断电后时间可能不准。若上面时间不对, 先校时:"
echo "    sudo timedatectl set-timezone Asia/Shanghai"
echo "    sudo timedatectl set-ntp true        # 联网自动同步"
echo "    sudo date -s \"2026-06-04 17:30:00\"   # 或手动设"
echo
echo "将要删除以下内容(不可恢复):"
echo "  • 数据库(账号/检测/报警/批次/审计全清): $HERE/db/tobacco.db"
echo "  • 产线生产配置:                          $HERE/db/line_config.json"
echo "  • Python 缓存:                           $HERE 下所有 __pycache__ / *.pyc"
echo "  • 报警证据帧:                            $HERE/logs/"
echo "  • 工人拍摄图:                            $CAPTURE_DIR/"
echo
read -r -p "确认全部清空、从头开始? 输入 yes 继续: " ANS
if [ "$ANS" != "yes" ]; then
  echo "已取消, 未做任何改动。"
  exit 0
fi

echo "→ 删数据库与产线配置..."
rm -f "$HERE/db/tobacco.db" "$HERE/db/line_config.json"

echo "→ 删 Python 缓存..."
find "$HERE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
find "$HERE" -name "*.pyc" -delete 2>/dev/null

echo "→ 删报警证据帧..."
rm -rf "$HERE/logs/"* 2>/dev/null

echo "→ 删工人拍摄图..."
rm -rf "$CAPTURE_DIR/"* 2>/dev/null

echo
echo "✅ 重置完成。下次启动将重建空库, 用 admin / admin123 登录(请尽快改密)。"
