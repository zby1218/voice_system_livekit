#!/usr/bin/env python3
"""
Face 唤醒编排：TCP 服务端接收 Face 的 present，拉起 run_client；会话结束后向 Face 发 resume。

先启动本脚本，再启动 Face（带 --notify-host --notify-port --resume-port）。

模块结构：
  FaceWakeConfig          - 运行参数配置（dataclass）
  CameraAvailabilityChecker - 轻量摄像头可用性检测（不依赖 face 目录）
  PresentEvent            - present 协议解析（值对象）
  PresentServer           - TCP 服务端，接收并解析 present 事件
  FaceWakeOrchestrator    - 主编排器：监听 → 会话 → resume
"""
import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _get_logger() -> logging.Logger:
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


# ---------------------------------------------------------------------------
# 协议常量
# ---------------------------------------------------------------------------

EVENT_PRESENT = "present"
EVENT_RESUME = "resume"

DEFAULT_PRESENT_PORT = 9999
DEFAULT_FACE_HOST = "127.0.0.1"
DEFAULT_FACE_RESUME_PORT = 9998
RESUME_RETRIES = 3
RESUME_RETRY_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# send_resume 工具函数（模块级，供外部直接调用）
# ---------------------------------------------------------------------------


def send_resume(
    host: str,
    port: int,
    retries: int = RESUME_RETRIES,
    retry_interval: float = RESUME_RETRY_INTERVAL,
) -> bool:
    """向 Face 的 resume 端口发送 resume，带重试。

    Args:
        host: Face 所在主机地址。
        port: Face 监听 resume 的端口。
        retries: 失败重试次数。
        retry_interval: 重试间隔（秒）。

    Returns:
        是否发送成功。
    """
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
    log.error("resume 在 %d 次重试后仍失败，face_detect 可能需要手动恢复", retries)
    return False


# ---------------------------------------------------------------------------
# FaceWakeConfig — 运行参数配置
# ---------------------------------------------------------------------------


@dataclass
class FaceWakeConfig:
    """face_wake_listener 的全部运行参数。"""

    present_port: int = DEFAULT_PRESENT_PORT
    face_host: str = DEFAULT_FACE_HOST
    face_resume_port: int = DEFAULT_FACE_RESUME_PORT
    livekit_retry_delay: float = 30.0
    check_camera: bool = False


# ---------------------------------------------------------------------------
# CameraAvailabilityChecker — 轻量摄像头检测
# ---------------------------------------------------------------------------


class CameraAvailabilityChecker:
    """轻量摄像头可用性检测，不依赖 face 目录，仅使用 cv2 与 sysfs。

    - Linux：先读 /sys/class/video4linux 枚举设备，再尝试 VideoCapture 打开。
    - 其他平台：依次尝试索引 0/1。
    探测期间抑制 OpenCV 日志，避免刷屏。
    """

    _V4L_DIR = "/sys/class/video4linux"

    def _usb_indices(self) -> list:
        """返回要探测的 USB 设备索引列表。"""
        if sys.platform.startswith("linux") and os.path.isdir(self._V4L_DIR):
            indices = []
            for name in os.listdir(self._V4L_DIR):
                suffix = name[5:]
                if name.startswith("video") and suffix.isdigit():
                    indices.append(int(suffix))
            if indices:
                return sorted(i for i in indices if i < 10)
        return [0, 1]

    def has_camera(self) -> bool:
        """检测是否有可用的 USB 摄像头。"""
        try:
            import cv2
        except ImportError:
            return False

        indices = self._usb_indices()
        try:
            save_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else 3
        except Exception:
            save_level = 3

        found = False
        try:
            cv2.setLogLevel(0)
            for idx in indices:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.release()
                    found = True
                    break
        finally:
            try:
                cv2.setLogLevel(save_level)
            except Exception:
                pass
        return found

    def check_or_exit(self, logger: logging.Logger) -> None:
        """无摄像头时打印错误并以 exit(1) 退出。"""
        if not self.has_camera():
            logger.error("未检测到摄像头，退出。请连接摄像头后重试。")
            sys.exit(1)
        logger.info("摄像头检测通过。")


# ---------------------------------------------------------------------------
# PresentEvent — present 协议解析
# ---------------------------------------------------------------------------


class PresentEvent:
    """解析来自 Face 的 present 事件（值对象，不可变）。"""

    def __init__(self, gender: Optional[str]):
        self._gender = gender

    @property
    def gender(self) -> Optional[str]:
        return self._gender

    @classmethod
    def parse(cls, raw_bytes: bytes) -> Optional["PresentEvent"]:
        """从原始字节解析 present 事件。失败或非 present 事件返回 None。"""
        try:
            line = raw_bytes.decode("utf-8", errors="ignore").strip()
            if not line:
                return None
            obj = json.loads(line)
            if obj.get("event") == EVENT_PRESENT:
                return cls(gender=obj.get("gender"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return None


# ---------------------------------------------------------------------------
# PresentServer — TCP 服务端，接收并解析 present 事件
# ---------------------------------------------------------------------------


class PresentServer:
    """TCP 服务端：绑定 present_port，每次 accept 一条连接并解析 present 事件。"""

    def __init__(self, port: int, logger: logging.Logger):
        self._port = port
        self._log = logger
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        """绑定端口并开始监听。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self._port))
        sock.listen(1)
        self._sock = sock
        self._log.info("监听 present 端口 %d，等待 Face 连接...", self._port)

    def accept_one(self) -> Optional[PresentEvent]:
        """阻塞等待一次连接，接收并返回 PresentEvent；解析失败返回 None。

        Raises:
            OSError: accept 失败时抛出，外层应捕获并决定是否退出。
        """
        if self._sock is None:
            raise RuntimeError("PresentServer 未启动，请先调用 start()")

        conn, addr = self._sock.accept()
        self._log.debug("收到连接: %s", addr)
        buf = b""
        with conn:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
        return PresentEvent.parse(buf)

    def close(self) -> None:
        """关闭服务端 socket。"""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ---------------------------------------------------------------------------
# FaceWakeOrchestrator — 主编排器
# ---------------------------------------------------------------------------


class FaceWakeOrchestrator:
    """主编排器：持续监听 present → 运行 LiveKit 会话 → 保证发送 resume。

    生命周期：
      1. 可选摄像头检测（默认不检测；需检测时传 --check-camera，无摄像头则退出）。
      2. 绑定 present 端口，阻塞等待 Face 发来 present。
      3. 收到 present → 调用 run_client 进行语音会话。
         - 正常完成：client_face 内部已发 resume，直接进入下一轮等待。
         - 异常（LiveKit 不可达等）：本编排器补发 resume，等待
           livekit_retry_delay 秒后再接受下一次 present，避免频繁重连。
      4. KeyboardInterrupt / 进程退出：补发 resume，确保 face_detect 不死锁。
    """

    def __init__(self, config: FaceWakeConfig):
        self._config = config
        self._log = _get_logger()
        self._server = PresentServer(config.present_port, self._log)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _send_resume(self) -> None:
        """向 Face 发 resume（带重试）。"""
        send_resume(self._config.face_host, self._config.face_resume_port)

    def _run_session(self, event: PresentEvent) -> None:
        """执行一次 LiveKit 会话。

        - 正常返回：run_client 内部已通过 data_received 事件发送了 resume，
          此处不重复发送。
        - 抛出异常：run_client 未能发送 resume，由此处补发，同时等待
          livekit_retry_delay 秒，防止频繁连接 LiveKit。
        - KeyboardInterrupt：补发 resume 后继续向上抛出，让进程正常退出。
        """
        from client_face import run_client

        cfg = self._config
        try:
            asyncio.run(
                run_client(
                    gender=event.gender,
                    face_resume_host=cfg.face_host,
                    face_resume_port=cfg.face_resume_port,
                )
            )
            # 正常完成：resume 已由 client_face 内部发出，无需补发
        except KeyboardInterrupt:
            self._log.info("会话被手动中断，补发 resume")
            self._send_resume()
            raise
        except Exception as e:
            self._log.exception("run_client 异常: %s", e)
            # 异常：client_face 未能发 resume，此处补发，避免 face_detect 死锁
            self._log.warning("会话异常，向 face_detect 补发 resume")
            self._send_resume()
            delay = cfg.livekit_retry_delay
            if delay > 0:
                self._log.warning(
                    "LiveKit 连接失败，等待 %.0f 秒后恢复监听...", delay
                )
                time.sleep(delay)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self) -> None:
        """启动主监听循环，阻塞直到进程退出。"""
        cfg = self._config
        log = self._log

        if cfg.check_camera:
            CameraAvailabilityChecker().check_or_exit(log)

        log.info(
            "会话结束后将向 %s:%d 发送 resume",
            cfg.face_host,
            cfg.face_resume_port,
        )
        self._server.start()
        try:
            while True:
                try:
                    event = self._server.accept_one()
                except OSError as e:
                    log.error("accept 异常，退出监听: %s", e)
                    break

                if event is None:
                    log.debug("收到无效数据，跳过")
                    continue

                log.info("收到 present (gender=%s)，启动 client...", event.gender)
                self._run_session(event)
                log.info("等待下一次 present...")
        finally:
            self._server.close()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face 唤醒编排：收 present 后跑 client，会话结束发 resume"
    )
    parser.add_argument(
        "--present-port",
        type=int,
        default=DEFAULT_PRESENT_PORT,
        help="本机监听 present 的端口（默认 9999）",
    )
    parser.add_argument(
        "--face-host",
        type=str,
        default=DEFAULT_FACE_HOST,
        help="Face 所在主机，用于发 resume（默认 127.0.0.1）",
    )
    parser.add_argument(
        "--resume-port",
        type=int,
        default=DEFAULT_FACE_RESUME_PORT,
        help="Face 监听 resume 的端口（默认 9998）",
    )
    parser.add_argument(
        "--livekit-retry-delay",
        type=float,
        default=30.0,
        help="LiveKit 连接失败后等待多少秒再继续监听（默认 30，0 表示不等待）",
    )
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help="启动时检测摄像头，无摄像头则退出（默认不检测：摄像头由 face_ratio_detect 占用时单独跑 listener 不会误报）",
    )
    parser.add_argument(
        "--no-check-camera",
        dest="check_camera",
        action="store_false",
        help="跳过摄像头检测（默认行为）",
    )
    args = parser.parse_args()

    config = FaceWakeConfig(
        present_port=args.present_port,
        face_host=args.face_host,
        face_resume_port=args.resume_port,
        livekit_retry_delay=args.livekit_retry_delay,
        check_camera=args.check_camera,
    )
    FaceWakeOrchestrator(config).run()


if __name__ == "__main__":
    main()
    sys.exit(0)
