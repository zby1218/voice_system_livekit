#!/usr/bin/env python3
"""
KWS 唤醒编排：TCP 服务端接收 project/kws 的 wake，拉起 client_kws 连 LiveKit；会话结束后回写 session_end。

与 face_wake_listener 对称。启动顺序（三选一理解）：
  - 先 kws_server → 再本脚本 → 再 project/kws/kws_trigger.py
  - 或：先本脚本 → 再 kws_server → 再 kws_trigger
  - 与 Face 一致：Listener 先起，检测端（face / kws_trigger）后起，检测到再连 Listener。
"""
import argparse
import asyncio
import json
import logging
import socket
import sys

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

EVENT_WAKE = "wake"
EVENT_SESSION_END = "session_end"

DEFAULT_WAKE_PORT = 9997


def _get_logger():
    logger = logging.getLogger("kws_wake_listener")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def run_listen_loop(wake_port: int):
    """阻塞循环：在 wake_port 上 accept，收 wake 后跑 run_client_kws，结束时回写 session_end。"""
    from client_kws import run_client

    log = _get_logger()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", wake_port))
    server.listen(1)
    log.info("监听 wake 端口 %d，等待 KWS 触发器连接...", wake_port)

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
            keyword = ""
            score = 0.0
            if line:
                try:
                    obj = json.loads(line)
                    if obj.get("event") == EVENT_WAKE:
                        keyword = obj.get("keyword", "")
                        score = float(obj.get("score", 0.0))
                        log.info("收到 wake (keyword=%s, score=%.3f)，启动 client_kws...", keyword, score)
                except json.JSONDecodeError:
                    pass

            try:
                asyncio.run(run_client(keyword=keyword))
            except KeyboardInterrupt:
                log.info("run_client 被中断")
                raise
            except Exception as e:
                log.exception("run_client 异常: %s", e)

            # 会话结束，回写 session_end 让 KWS 触发器解除阻塞
            try:
                reply = json.dumps({"event": EVENT_SESSION_END}, ensure_ascii=False) + "\n"
                conn.sendall(reply.encode("utf-8"))
            except (socket.error, OSError):
                pass

        log.info("等待下一次 wake...")

    server.close()


def main():
    parser = argparse.ArgumentParser(description="KWS 唤醒编排：收 wake 后跑 client_kws，会话结束回写 session_end")
    parser.add_argument("--wake-port", type=int, default=DEFAULT_WAKE_PORT, help="本机监听 wake 的端口")
    args = parser.parse_args()
    run_listen_loop(args.wake_port)


if __name__ == "__main__":
    main()
    sys.exit(0)
