#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人分离本地服务：加载 MossFormer2_SS_16K，接收 PCM 音频字节，返回第一路说话人 PCM。

用法：在项目根目录执行
  python -m speaker_select.server [--host 0.0.0.0] [--port 8000]

stt_server_novad 在 ENABLE_SPEAKER_SELECT=True 时，将音频 POST 到本服务的 /separate，取回分离后的音频再送 ASR。
"""

import argparse
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# 支持直接运行 python speaker_select/server.py：把项目根加入 path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from speaker_select.separator import SpeakerSeparator

logger = logging.getLogger("SpeakerSelectService")

SEPARATE_PATH = "/separate"


class SeparateHandler(BaseHTTPRequestHandler):
    """POST /separate：请求体为 PCM s16le 16kHz 单声道字节，响应体为第一路说话人 PCM 字节。"""

    def do_POST(self):
        if self.path != SEPARATE_PATH:
            self.send_error(404, "Not Found")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self.send_error(400, "Content-Length required")
            return
        try:
            body = self.rfile.read(content_length)
        except Exception as e:
            logger.exception("读取请求体失败: %s", e)
            self.send_error(500, str(e))
            return
        separator = getattr(self.server, "separator", None)
        if separator is None:
            self.send_error(503, "Separator not loaded")
            return
        try:
            speakers = separator._separate_sync(body)
            first = speakers[0] if speakers else b""
        except Exception as e:
            logger.exception("分离失败: %s", e)
            self.send_error(500, str(e))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", len(first))
        self.end_headers()
        self.wfile.write(first)

    def log_message(self, format, *args):
        logger.info(format % args)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="说话人分离本地服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=1999, help="监听端口")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="推理设备")
    args = parser.parse_args()

    logger.info("正在加载 MossFormer2_SS_16K（device=%s）...", args.device)
    separator = SpeakerSeparator(device=args.device)
    logger.info("模型加载完成，启动 HTTP 服务 %s:%s", args.host, args.port)

    server = HTTPServer((args.host, args.port), SeparateHandler)
    server.separator = separator
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到退出信号，停止服务")
        server.shutdown()


if __name__ == "__main__":
    main()
