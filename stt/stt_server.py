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

# 需要先引入FunASRNano，不然会报错：FunASRNano is not registered
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
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        model_root = os.path.join(current_dir, "model")
        
        logger.info(f"锁定模型根目录: {model_root}")

        # === 2. 拼接具体模型路径 ===
        # ASR: model/FunAudioLLM/Fun-ASR-Nano-2512
        self.path_asr = os.path.join(model_root, "FunAudioLLM", "Fun-ASR-Nano-2512")
        
        # VAD: model/models/iic/speech_fsmn_vad...
        self.path_vad = os.path.join(model_root, "models", "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
        
        # Punc: model/models/iic/punc_ct...
        self.path_punc = os.path.join(model_root, "models", "iic", "punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727")

        # 检查路径是否存在
        self._check_path(self.path_asr, "ASR")
        self._check_path(self.path_vad, "VAD")
        self._check_path(self.path_punc, "标点")

        # 模型实例对象
        self.model_asr = None
        self.model_vad = None
        self.model_punc = None
        
        self.websocket_users = set()
        self.running = False
        self.server = None

    def _check_path(self, path, name):
        if not os.path.exists(path):
            logger.error(f"❌ 找不到 {name} 模型路径: {path}")
            logger.error(f"请检查文件夹结构，预期路径为: {path}")
            sys.exit(1)
        else:
            logger.info(f"✅ {name} 路径确认: {path}")

    def load_models(self):
        logger.info(f"开始加载模型 (Device: {self.device})...")

        try:
            # 加载 VAD
            # 修正点：使用 self.path_vad 而不是 self.model_id_vad
            self.model_vad = AutoModel(
                model=self.path_vad,
                device=self.device,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
                local_files_only=True
            )

            # 加载 ASR
            # 修正点：使用 self.path_asr
            self.model_asr = AutoModel(
                model=self.path_asr,
                device=self.device,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
                local_files_only=True
            )

            # 加载 Punc
            # 修正点：使用 self.path_punc
            self.model_punc = AutoModel(
                model=self.path_punc,
                device=self.device,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
                local_files_only=True
            )
            logger.info("🎉 所有模型加载成功！服务准备就绪。")

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
        websocket.status_dict_vad = {"cache": {}, "is_final": False}
        websocket.status_dict_punc = {"cache": {}}
        websocket.chunk_interval = 10
        websocket.mode = "2pass"
        websocket.is_speaking = True
        websocket.wav_name = "mic"
        websocket.vad_pre_idx = 0
        
        # 缓冲区
        frames = []            # 全部帧（用于回溯）
        frames_asr = []        # 人声音频帧
        frames_asr_online = [] # 流式识别缓冲
        
        # VAD 状态
        speech_start = False
        speech_end_i = -1
        
        logger.info(f"新客户端连接: {websocket.remote_address}")

        try:
            async for message in websocket:
                # === 1. 处理 JSON 配置 ===
                if isinstance(message, str):
                    try:
                        msg_json = json.loads(message)
                        
                        if "chunk_size" in msg_json:
                            chunk = msg_json["chunk_size"]
                            if isinstance(chunk, str): chunk = [int(x) for x in chunk.split(",")]
                            websocket.status_dict_asr_online["chunk_size"] = chunk
                            websocket.status_dict_vad["chunk_size"] = int(chunk[1] * 60 / websocket.chunk_interval)
                        
                        if "mode" in msg_json:
                            websocket.mode = msg_json["mode"]
                        
                        if "wav_name" in msg_json:
                            websocket.wav_name = msg_json["wav_name"]

                        # 客户端主动结束信号
                        if "is_speaking" in msg_json:
                            websocket.is_speaking = msg_json["is_speaking"]
                            websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
                            
                            if not websocket.is_speaking:
                                logger.info(f"收到客户端结束信号，触发最终识别... 缓冲帧数: {len(frames_asr)}")
                                audio_in = b"".join(frames_asr)
                                await self.async_asr_offline(websocket, audio_in)
                                
                                # 重置所有状态
                                frames = []
                                frames_asr = []
                                frames_asr_online = []
                                speech_start = False
                                speech_end_i = -1
                                websocket.vad_pre_idx = 0
                                websocket.status_dict_asr_online["cache"] = {}
                                websocket.status_dict_vad["cache"] = {}

                    except Exception as e:
                        logger.error(f"JSON解析或处理错误: {e}")
                    
                    continue

                # === 2. 处理音频数据 (Bytes) ===
                frames.append(message)
                duration_ms = len(message) // 32  # 16kHz 16-bit = 32 bytes/ms
                websocket.vad_pre_idx += duration_ms
                
                # 2.1 流式识别缓冲
                frames_asr_online.append(message)
                websocket.status_dict_asr_online["is_final"] = (speech_end_i != -1)
                
                # 如果已检测到语音开始，累积音频
                if speech_start:
                    frames_asr.append(message)
                
                # 2.2 VAD 检测
                try:
                    speech_start_i, speech_end_i = await self.async_vad(websocket, message)
                except Exception as e:
                    logger.error(f"VAD 错误: {e}")
                    speech_start_i, speech_end_i = -1, -1
                
                # 处理语音开始
                if speech_start_i != -1:
                    speech_start = True
                    # 回溯获取语音开始前的帧
                    beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                    beg_bias = max(1, min(beg_bias, len(frames)))
                    frames_pre = frames[-beg_bias:] if beg_bias > 0 else []
                    frames_asr = list(frames_pre)
                    logger.info(f"🎤 检测到语音开始, 回溯 {len(frames_pre)} 帧")
                
                # 2.3 流式识别
                if len(frames_asr_online) > 0 and (len(frames_asr_online) % websocket.chunk_interval == 0 or speech_end_i != -1):
                    audio_in = b"".join(frames_asr_online)
                    await self.async_asr_online(websocket, audio_in)
                    frames_asr_online = []
                
                # 2.4 语音结束 -> 触发离线识别
                if speech_end_i != -1:
                    logger.info(f"🔇 检测到语音结束, 触发离线识别... 帧数: {len(frames_asr)}")
                    audio_in = b"".join(frames_asr)
                    if len(audio_in) > 0:
                        await self.async_asr_offline(websocket, audio_in)
                    
                    # 重置状态
                    frames_asr = []
                    frames_asr_online = []
                    speech_start = False
                    speech_end_i = -1
                    websocket.vad_pre_idx = 0
                    frames = []
                    websocket.status_dict_asr_online["cache"] = {}
                    websocket.status_dict_vad["cache"] = {}

        except websockets.ConnectionClosed:
            logger.info("客户端断开连接")
        except Exception as e:
            logger.error(f"处理异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.websocket_users.discard(websocket)

    async def async_vad(self, websocket, audio_in):
        """异步 VAD 检测语音端点"""
        audio_tensor = self.decode_audio_chunk(audio_in)
        segments_result_list = await self.run_model_inference(
            self.model_vad, input=[audio_tensor], **websocket.status_dict_vad
        )
        if not segments_result_list or len(segments_result_list) == 0:
            return -1, -1
        
        segments_result = segments_result_list[0]["value"]
        speech_start = -1
        speech_end = -1
        
        if len(segments_result) == 0 or len(segments_result) > 1:
            return speech_start, speech_end
        
        if segments_result[0][0] != -1:
            speech_start = segments_result[0][0]
        if segments_result[0][1] != -1:
            speech_end = segments_result[0][1]
        
        return speech_start, speech_end

    async def async_asr_online(self, websocket, audio_in):
        if len(audio_in) == 0: return
        try:
            audio_tensor = self.decode_audio_chunk(audio_in)
            res = await self.run_model_inference(
                self.model_asr, input=[audio_tensor], **websocket.status_dict_asr_online
            )
            text = res[0]['text'] if res else ""
            if text:
                await websocket.send(json.dumps({
                    "mode": "2pass-online", "text": text, "is_final": False
                }))
        except Exception:
            pass

    async def async_asr_offline(self, websocket, audio_in):
        if len(audio_in) == 0: return
        try:
            audio_tensor = self.decode_audio_chunk(audio_in)
            res = await self.run_model_inference(self.model_asr, input=[audio_tensor])
            text = res[0]['text'] if res else ""
            
            if self.model_punc and text:
                res_punc = await self.run_model_inference(self.model_punc, input=text)
                if res_punc:
                    # 标点模型返回的格式可能是字典或字符串
                    punc_result = res_punc[0]
                    if isinstance(punc_result, dict):
                        text = punc_result.get('text', text)
                    elif isinstance(punc_result, str):
                        text = punc_result
                    else:
                        # 如果是 Tensor 或其他类型，转换为字符串
                        text = str(punc_result) if punc_result else text
            print(text)
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