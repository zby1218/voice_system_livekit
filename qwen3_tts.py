import torch
import pyaudio
import numpy as np
import soundfile as sf
import time
import threading
import queue
import re
from qwen_tts import Qwen3TTSModel
from openai import OpenAI

# ==================== 1. 全局配置与模型初始化 ====================

MODEL_PATH = "/home/zhangchi/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LLM_API_KEY = "sk-3a9ca0038afe434abf0c703000ebb694"
SAMPLE_RATE = 24000  # Qwen3-TTS 固定采样率

print("⏳ 正在加载 TTS 模型...")
tts_model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print("✅ TTS 模型加载完毕")

llm_client = OpenAI(
    api_key=LLM_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ==================== 2. PyAudio 播放器类 ====================

class PyAudioTTSPlayer:
    def __init__(self, model, sample_rate=24000, seed=1234): # <--- 【修改点1】传入 seed
        self.model = model
        self.sample_rate = sample_rate
        self.seed = seed  # 保存种子
        
        # 队列用于线程间通信
        self.text_queue = queue.Queue(maxsize=20)
        self.audio_queue = queue.Queue(maxsize=20)
        
        # 状态标志
        self.is_running = True
        
        # 数据统计
        self.full_audio_buffer = [] # 存储所有音频用于保存
        self.stats = {
            "total_gen_time": 0.0,
            "total_duration": 0.0
        }

        # === PyAudio 初始化核心部分 ===
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,  # 必须是 Float32，因为模型输出是 float
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        print(f"🔊 PyAudio 音频流已开启 (Rate: {self.sample_rate}Hz, Format: Float32)")
        print(f"🎲 当前随机种子 (Seed): {self.seed}")

    def _generate_worker(self, language, instruct):
        """生成线程：负责推理和计算 RTF"""
        print("🎙️ 生成线程启动...")
        chunk_idx = 0
        
        while True:
            text = self.text_queue.get()
            if text is None: # 结束信号
                break
                
            chunk_idx += 1
            print(f"\nProcessing Chunk {chunk_idx}: 「{text[:15]}...」")
            
            # --- RTF 计时开始 ---
            t_start = time.time()
            
            try:
                # =================================================
                # 【修改点2】每次生成前，重置随机种子
                # 确保每一句话都从相同的随机状态开始，保证音色一致
                # =================================================
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                torch.manual_seed(self.seed)

                # 调用模型生成 (参数组合建议：Seed固定 + Temp适中 + Top_P适中 + RepPenalty)
                wavs, sr = self.model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct,
                    do_sample=True,          # 必须开启采样
                    top_p=0.8,               # 过滤低概率token
                    temperature=0.7,         # 控制随机性，配合seed使用
                    repetition_penalty=1.1,  # 防止复读
                    max_new_tokens=2048
                )
                audio_data = wavs[0]
            except Exception as e:
                print(f"❌ 生成错误: {e}")
                import traceback
                traceback.print_exc()
                continue
                
            # --- RTF 计时结束 ---
            t_end = time.time()
            
            # 计算统计数据
            gen_time = t_end - t_start
            audio_dur = len(audio_data) / sr
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            
            # 更新全局统计
            self.stats["total_gen_time"] += gen_time
            self.stats["total_duration"] += audio_dur
            self.full_audio_buffer.append(audio_data) # 存入全量 buffer
            
            print(f"   ⚡ RTF: {rtf:.3f} (生成 {gen_time:.2f}s / 音频 {audio_dur:.2f}s)")
            if rtf > 1.0:
                print("   ⚠️ 警告: 生成速度慢于播放速度")

            # 将数据放入播放队列
            self.audio_queue.put(audio_data)
        
        # 发送播放结束信号
        self.audio_queue.put(None)
        print("✅ 生成线程结束")

    def _play_worker(self):
        """播放线程：使用 PyAudio 写入数据"""
        print("🎧 播放线程启动...")
        
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None: # 结束信号
                break
            
            # === PyAudio 播放核心 ===
            # model 输出是 np.float32，直接转 bytes 写入 stream
            # 如果不想听起来像快进，不需要手动 sleep，stream.write 是阻塞的
            self.stream.write(audio_data.tobytes())
            
        print("✅ 播放线程结束")

    def start(self, user_input, language="Chinese", instruct=""):
        """启动整个流程"""
        # 1. 启动 LLM 线程 (生产者 1)
        t_llm = threading.Thread(target=self._stream_llm, args=(user_input,))
        
        # 2. 启动 TTS 生成线程 (生产者 2 / 消费者 1)
        t_gen = threading.Thread(target=self._generate_worker, args=(language, instruct))
        
        # 3. 启动 播放线程 (消费者 2)
        t_play = threading.Thread(target=self._play_worker)
        
        t_play.start()
        t_gen.start()
        t_llm.start()
        
        # 等待所有线程结束
        t_llm.join()
        t_gen.join()
        t_play.join()
        
        self.cleanup()
        self.print_stats()

    def _stream_llm(self, user_input):
        """LLM 流式处理与分句 (带积攒逻辑)"""
        buffer = ""
        strong_delimiters = r'[。！？\n]'
        weak_delimiters = r'[，；]'
        
        messages = [{"role": "system", "content": "回答纯净的文本，不要带表情符号及markdown格式，同时回答的长度适中，不要太长也不要太短"}, {"role": "user", "content": user_input}]
        completion = llm_client.chat.completions.create(
            model="qwen-plus-2025-12-01",
            messages=messages,
            stream=True,
            extra_body={"enable_thinking": False}
        )
        
        print("--- LLM Output ---")
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                text = delta.content
                print(text, end="", flush=True)
                buffer += text
                
                # 分句逻辑：遇到强标点切分
                parts = re.split(f'({strong_delimiters})', buffer)
                if len(parts) > 1:
                    for i in range(0, len(parts) - 1, 2):
                        sentence = parts[i] + parts[i+1]
                        if len(sentence.strip()) > 1:
                            self.text_queue.put(sentence.strip())
                    buffer = parts[-1]
                # 兜底逻辑：太长强制切
                elif len(buffer) > 80:
                    parts = re.split(f'({weak_delimiters})', buffer)
                    if len(parts) > 1:
                        self.text_queue.put((parts[0] + parts[1]).strip())
                        buffer = "".join(parts[2:])
                        
        if buffer.strip():
            self.text_queue.put(buffer.strip())
        
        self.text_queue.put(None) # LLM 结束
        print("\n------------------")

    def save_wav(self, filename="output.wav"):
        """保存全量音频"""
        if not self.full_audio_buffer:
            print("⚠️ 没有音频数据可保存")
            return
        
        print(f"💾 正在保存到 {filename} ...")
        full_data = np.concatenate(self.full_audio_buffer)
        sf.write(filename, full_data, self.sample_rate)
        print("✅ 保存成功")

    def print_stats(self):
        """打印最终统计"""
        if self.stats["total_duration"] > 0:
            avg_rtf = self.stats["total_gen_time"] / self.stats["total_duration"]
            print("\n" + "="*30)
            print(f"📊 统计报告:")
            print(f"   总音频时长: {self.stats['total_duration']:.2f}s")
            print(f"   总推理耗时: {self.stats['total_gen_time']:.2f}s")
            print(f"   平均 RTF   : {avg_rtf:.3f}")
            print("="*30 + "\n")

    def cleanup(self):
        """释放 PyAudio 资源"""
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("🔌 资源已释放")

# ==================== 3. 主程序入口 ====================

if __name__ == "__main__":
    # 【修改点3】在这里传入你想要的 Seed (整数)
    # 比如: 42, 10086, 9999
    # 不同的 seed 对应不同的音色
    player = PyAudioTTSPlayer(tts_model, seed=404) 
    
    # 示例 prompt
    prompt = "请用这是一种带有磁性的男声，语气沉稳，语速和日常说话聊天一致。"
    text_input = "介绍一下北京市。"
    
    # 开始对话
    player.start(text_input, instruct=prompt)
    
    # 保存结果
    player.save_wav("chat_history_with_seed.wav")