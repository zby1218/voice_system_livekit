#!/usr/bin/env python3
"""
Face 唤醒编排：TCP 服务端接收 Face 的 present，拉起 run_client；会话结束后向 Face 发 resume。
先启动本脚本，再启动 Face（带 --notify-host --notify-port --resume-port）。
"""
import argparse
import asyncio
import json
import logging
import socket
import sys
import time

# 日志格式与 face 侧一致：时间 + 级别 + 名称 + 消息
_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _get_logger():
    logger = logging.getLogger("face_wake_listener")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

# 协议与 Face 侧一致：一行 JSON
EVENT_PRESENT = "present"
EVENT_RESUME = "resume"

DEFAULT_PRESENT_PORT = 9999
DEFAULT_FACE_HOST = "127.0.0.1"
DEFAULT_FACE_RESUME_PORT = 9998
RESUME_RETRIES = 3
RESUME_RETRY_INTERVAL = 1.0


def send_resume(host: str, port: int, retries: int = RESUME_RETRIES, retry_interval: float = RESUME_RETRY_INTERVAL) -> bool:
    """向 Face 的 resume 端口发送 resume，带重试。"""
    log = _get_logger()
    payload = {"event": EVENT_RESUME}
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    data = line.encode("utf-8")
    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((host, port), timeout=5.0) as s:
                s.sendall(data)
            log.info("已发送 resume 到 %s:%d", host, port)
            return True
        except (socket.error, OSError) as e:
            log.warning("发送 resume 失败 (尝试 %d/%d): %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(retry_interval)
    return False


def run_listen_loop(present_port: int, face_host: str, face_resume_port: int):
    """阻塞循环：在 present_port 上 accept，收 present 后跑 run_client，再发 resume。"""
    from client_face import run_client

    log = _get_logger()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", present_port))
    server.listen(1)
    log.info("监听 present 端口 %d，等待 Face 连接...", present_port)
    log.info("会话结束后将向 %s:%d 发送 resume", face_host, face_resume_port)

    while True:
        try:
            conn, addr = server.accept()
        except OSError as e:
            log.error("accept 异常: %s", e)
            break
        with conn:
            buf = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            line = buf.decode("utf-8", errors="ignore").strip()
            gender = None
            if line:
                try:
                    obj = json.loads(line)
                    if obj.get("event") == EVENT_PRESENT:
                        gender = obj.get("gender")
                        log.info("收到 present (gender=%s)，启动 client...", gender)
                except json.JSONDecodeError:
                    pass

        try:
            asyncio.run(run_client(
                gender=gender,
                face_resume_host=face_host,
                face_resume_port=face_resume_port,
            ))
        except KeyboardInterrupt:
            log.info("run_client 被中断")
            raise
        except Exception as e:
            log.exception("run_client 异常: %s", e)

        log.info("等待下一次 present...")

    server.close()


def main():
    parser = argparse.ArgumentParser(description="Face 唤醒编排：收 present 后跑 client，会话结束发 resume")
    parser.add_argument("--present-port", type=int, default=DEFAULT_PRESENT_PORT, help="本机监听 present 的端口")
    parser.add_argument("--face-host", type=str, default=DEFAULT_FACE_HOST, help="Face 所在主机（发 resume 用）")
    parser.add_argument("--resume-port", type=int, default=DEFAULT_FACE_RESUME_PORT, help="Face 监听 resume 的端口")
    args = parser.parse_args()
    run_listen_loop(args.present_port, args.face_host, args.resume_port)


if __name__ == "__main__":
    main()
    sys.exit(0)
