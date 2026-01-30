#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import sys
import json
import asyncio
import logging
import websockets
import torch
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor

# 忽略警告
warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ASRServer")

# 抑制 transformers 的 attention_mask / pad_token_id 警告
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

# 需要先引入FunASRNano
try:
    from funasr.models.fun_asr_nano.model import FunASRNano
    from funasr import AutoModel
except ImportError:
    print("❌ 错误: 未安装 funasr。请运行: pip install funasr torch websockets numpy")
    sys.exit(1)

# 全局线程池
inference_executor = ThreadPoolExecutor(max_workers=5)

class ASRServer:
    def __init__(self, host="0.0.0.0", port=10095, device="cuda"):
        self.host = host
        self.port = port
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # === 1. 路径设置 ===
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_root = os.path.join(current_dir, "model")
        
        logger.info(f"锁定模型根目录: {model_root}")

        # === 2. 拼接具体模型路径 ===
        # ASR: model/FunAudioLLM/Fun-ASR-Nano-2512
        self.path_asr = os.path.join(model_root, "FunAudioLLM", "Fun-ASR-Nano-2512")
        
        # Punc: model/models/iic/punc_ct...
        # self.path_punc = os.path.join(model_root, "models", "iic", "punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727")

        # 检查路径是否存在 (已移除 VAD 检查)
        self._check_path(self.path_asr, "ASR")
        # self._check_path(self.path_punc, "标点")

        # 模型实例对象
        self.model_asr = None
        # self.model_punc = None
        
        self.websocket_users = set()
        self.running = False
        self.server = None

    def _check_path(self, path, name):
        if not os.path.exists(path):
            logger.error(f"❌ 找不到 {name} 模型路径: {path}")
            sys.exit(1)
        else:
            logger.info(f"✅ {name} 路径确认: {path}")

    def load_models(self):
        logger.info(f"开始加载模型 (Device: {self.device})...")

        try:
            # 已移除 VAD 加载逻辑

            # 加载 ASR
            self.model_asr = AutoModel(
                model=self.path_asr,
                device=self.device,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
                local_files_only=True
            )

            # 加载 Punc
            # self.model_punc = AutoModel(
            #     model=self.path_punc,
            #     device=self.device,
            #     disable_pbar=True,
            #     disable_log=True,
            #     disable_update=True,
            #     local_files_only=True
            # )
            logger.info("🎉 模型加载成功！服务准备就绪。")

        except Exception as e:
            logger.error(f"❌ 模型加载崩溃: {e}")
            sys.exit(1)

    async def start(self):
        self.load_models()
        self.server = await websockets.serve(
            self.ws_serve, self.host, self.port, subprotocols=None, ping_interval=None
        )
        self.running = True
        logger.info(f"🚀 服务已启动，监听地址: ws://{self.host}:{self.port}")
        await asyncio.Future()

    def decode_audio_chunk(self, chunk_bytes):
        # 注意：这里默认假设客户端传来的是 PCM s16le, 16000Hz
        data_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        data_float32 = data_int16.astype(np.float32) / 32768.0
        return torch.from_numpy(data_float32)

    async def run_model_inference(self, model, input, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            inference_executor,
            lambda: model.generate(input=input, **kwargs)
        )

    async def ws_serve(self, websocket, path=None):
        self.websocket_users.add(websocket)
        
        # 初始化状态
        websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
        websocket.chunk_interval = 10
        websocket.mode = "2pass"
        websocket.is_speaking = True
        
        # 缓冲区
        frames_asr = []        # 存储整句音频 (用于离线识别)
        frames_asr_online = [] # 存储流式片段 (用于实时出字)
        
        # 调试：消息计数
        msg_count = {"json": 0, "bytes": 0}
        
        logger.info(f"新客户端连接: {websocket.remote_address}")

        try:
            async for message in websocket:
                # === 1. 处理 JSON 配置 ===
                if isinstance(message, str):
                    msg_count["json"] += 1
                    try:
                        msg_json = json.loads(message)
                        
                        if "chunk_size" in msg_json:
                            chunk = msg_json["chunk_size"]
                            if isinstance(chunk, str): chunk = [int(x) for x in chunk.split(",")]
                            websocket.status_dict_asr_online["chunk_size"] = chunk
                        
                        if "chunk_interval" in msg_json:
                            websocket.chunk_interval = int(msg_json["chunk_interval"])
                            logger.info(f"📋 收到客户端配置 chunk_interval={websocket.chunk_interval}，等待音频...")
                        
                        if "mode" in msg_json:
                            websocket.mode = msg_json["mode"]

                        # === 核心修改：完全依赖客户端的 is_speaking 信号 ===
                        if "is_speaking" in msg_json:
                            websocket.is_speaking = msg_json["is_speaking"]
                            websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
                            
                            # 只有当客户端明确说“话说完了” (is_speaking=False)
                            # 服务端才进行最终的 Offline 高精度识别
                            if not websocket.is_speaking:
                                logger.info(f"📤 收到结束信号 is_speaking=False，触发 Final 识别... 总帧数: {len(frames_asr)}")
                                
                                # 只有缓冲区有数据才跑
                                if len(frames_asr) > 0:
                                    audio_in = b"".join(frames_asr)
                                    await self.async_asr_offline(websocket, audio_in)
                                
                                # 重置所有状态，准备下一句话
                                frames_asr = []
                                frames_asr_online = []
                                websocket.status_dict_asr_online["cache"] = {}

                    except Exception as e:
                        logger.error(f"JSON处理错误: {e}")
                    
                    continue

                # === 2. 处理音频数据 (Bytes) ===
                # 简单粗暴：来什么存什么
                msg_count["bytes"] += 1
                if msg_count["bytes"] <= 3 or msg_count["bytes"] % 50 == 0:
                    logger.info(f"📥 收到音频 chunk #{msg_count['bytes']} (len={len(message)} bytes)")
                
                frames_asr.append(message)
                frames_asr_online.append(message)
                
                # 2.1 流式识别 (Interim Results)
                # 每积累一定数据量（例如 10 个 chunk），跑一次 Online 模型
                if len(frames_asr_online) >= websocket.chunk_interval:
                    audio_in = b"".join(frames_asr_online)
                    logger.info(f"🎤 触发 Online 推理 (chunks={len(frames_asr_online)}, 总bytes={len(audio_in)})")
                    await self.async_asr_online(websocket, audio_in)
                    frames_asr_online = [] # 清空流式缓冲，避免重复计算

        except websockets.ConnectionClosed:
            logger.info(f"客户端断开连接 (本次会话: JSON={msg_count['json']}, 音频chunks={msg_count['bytes']})")
        except Exception as e:
            logger.error(f"处理异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.websocket_users.discard(websocket)

    async def async_asr_online(self, websocket, audio_in):
        if len(audio_in) == 0: return
        try:
            audio_tensor = self.decode_audio_chunk(audio_in)
            res = await self.run_model_inference(
                self.model_asr, input=[audio_tensor], **websocket.status_dict_asr_online
            )
            text = res[0]['text'] if res else ""
            if text:
                # 2pass-online 表示中间结果
                await websocket.send(json.dumps({
                    "mode": "2pass-online", "text": text, "is_final": False
                }))
        except Exception as e:
            logger.warning(f"Online 推理异常: {e}")

    async def async_asr_offline(self, websocket, audio_in):
        if len(audio_in) == 0: return
        try:
            audio_tensor = self.decode_audio_chunk(audio_in)
            # 离线识别不需要 cache
            res = await self.run_model_inference(self.model_asr, input=[audio_tensor])
            text = res[0]['text'] if res else ""
            
            # 标点预测
            # if self.model_punc and text:
            #     res_punc = await self.run_model_inference(self.model_punc, input=text)
            #     if res_punc:
            #         punc_result = res_punc[0]
            #         if isinstance(punc_result, dict):
            #             text = punc_result.get('text', text)
            #         elif isinstance(punc_result, str):
            #             text = punc_result
            #         else:
            #             text = str(punc_result) if punc_result else text
            
            logger.info(f"\n\n Final Result: {text} \n\n")
            
            # 2pass-offline 表示最终定稿结果
            await websocket.send(json.dumps({
                "mode": "2pass-offline", "text": text, "is_final": True
            }))
        except Exception as e:
            logger.error(f"离线识别错误: {e}")

if __name__ == "__main__":
    server = ASRServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("停止服务")