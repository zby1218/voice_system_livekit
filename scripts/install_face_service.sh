#!/usr/bin/env bash
# 安装/卸载 robot-face systemd 开机自启服务
#
# 用法：
#   sudo bash install_face_service.sh install    # 安装并启用服务
#   sudo bash install_face_service.sh uninstall  # 停止并删除服务
#   sudo bash install_face_service.sh status     # 查看服务状态
#   sudo bash install_face_service.sh logs       # 查看近期日志
#
# 本脚本会自动以 sudo 调用者的实际用户名替换服务文件中的 __DEPLOY_USER__ 占位符，
# 因此不需要手动修改任何配置文件。

set -euo pipefail

SERVICE_NAME="robot-face"
SERVICE_FILE_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/systemd/${SERVICE_NAME}.service"
SERVICE_FILE_DST="/etc/systemd/system/${SERVICE_NAME}.service"
SCRIPT_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_face.sh"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
info()  { echo "[install] $*"; }
warn()  { echo "[install] WARNING: $*" >&2; }
error() { echo "[install] ERROR: $*" >&2; exit 1; }

check_root() {
    [ "$(id -u)" -eq 0 ] || error "请以 root 权限运行（sudo bash $0 $1）"
}

check_files() {
    [ -f "$SERVICE_FILE_SRC" ] || error "未找到服务文件: $SERVICE_FILE_SRC"
    [ -f "$SCRIPT_FILE" ]      || error "未找到启动脚本: $SCRIPT_FILE"
}

# 获取真实调用者用户名（sudo 环境下 SUDO_USER 为实际用户）
get_deploy_user() {
    if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        echo "$SUDO_USER"
    else
        id -un
    fi
}

# 获取该用户的 home 目录（用于替换 __DEPLOY_HOME__，避免 %h 在 systemd 中展开为 /root）
get_deploy_home() {
    local user="$1"
    getent passwd "$user" 2>/dev/null | cut -d: -f6 || echo "$HOME"
}

# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
cmd_install() {
    check_root install
    check_files

    local deploy_user deploy_home
    deploy_user=$(get_deploy_user)
    deploy_home=$(get_deploy_home "$deploy_user")
    info "部署用户: $deploy_user (HOME=$deploy_home)"

    info "生成服务文件（替换占位符）: $SERVICE_FILE_DST"
    sed -e "s/__DEPLOY_USER__/$deploy_user/g" \
        -e "s|__DEPLOY_HOME__|$deploy_home|g" \
        -e "s|__SCRIPT_PATH__|$SCRIPT_FILE|g" \
        "$SERVICE_FILE_SRC" > "$SERVICE_FILE_DST"
    chmod 0644 "$SERVICE_FILE_DST"

    info "设置启动脚本可执行权限"
    chmod +x "$SCRIPT_FILE"

    info "重载 systemd 配置"
    systemctl daemon-reload

    info "启用开机自启"
    systemctl enable "$SERVICE_NAME"

    info "立即启动服务"
    systemctl start "$SERVICE_NAME"

    echo ""
    info "===== 安装完成 ====="
    info "查看状态:  sudo systemctl status $SERVICE_NAME"
    info "查看日志:  journalctl -u $SERVICE_NAME -f"
    info "手动停止:  sudo systemctl stop $SERVICE_NAME"
    info "禁用自启:  sudo systemctl disable $SERVICE_NAME"
}

cmd_uninstall() {
    check_root uninstall

    info "停止服务..."
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true

    info "禁用开机自启..."
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true

    if [ -f "$SERVICE_FILE_DST" ]; then
        info "删除服务文件: $SERVICE_FILE_DST"
        rm -f "$SERVICE_FILE_DST"
    fi

    info "重载 systemd 配置"
    systemctl daemon-reload

    info "===== 卸载完成 ====="
}

cmd_status() {
    systemctl status "$SERVICE_NAME" --no-pager || true
}

cmd_logs() {
    journalctl -u "$SERVICE_NAME" -n 100 --no-pager
}

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
ACTION="${1:-help}"
case "$ACTION" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    status)    cmd_status ;;
    logs)      cmd_logs ;;
    *)
        echo "用法: sudo bash $0 {install|uninstall|status|logs}"
        echo ""
        echo "  install    安装并启用 $SERVICE_NAME 开机自启服务"
        echo "  uninstall  停止并删除服务"
        echo "  status     查看服务当前状态"
        echo "  logs       查看近期服务日志"
        exit 1
        ;;
esac
