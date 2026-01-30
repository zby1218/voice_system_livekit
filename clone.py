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

# ==================== 1. 全局配置与模型加载 ====================

MODEL_PATH_DESIGN = "/home/zhangchi/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_PATH_BASE = "/home/zhangchi/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"

LLM_API_KEY = "sk-3a9ca0038afe434abf0c703000ebb694"
SAMPLE_RATE = 24000 

# 初始化 LLM 客户端
llm_client = OpenAI(
    api_key=LLM_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

print("🚀 [1/2] 正在加载 VoiceDesign 模型 (用于生成音色)...")
design_model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH_DESIGN,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

print("🚀 [2/2] 正在加载 Base 模型 (用于流式克隆)...")
base_model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH_BASE,
    device_map="cuda:0", # 如果显存够，放同一张卡推理最快；不够可以改 cuda:1
    dtype=torch.bfloat16,
)
print("✅ 双模型加载完毕！")


# ==================== 2. 音色预处理逻辑 ====================

def create_stable_prompt(text, instruct, seed=42):
    """
    使用 Design 模型生成参考音频，然后转换为 Base 模型可用的 Prompt
    """
    print(f"\n🎨 [Setup] 正在定制音色...")
    print(f"   - 设定: {instruct}")
    print(f"   - 参考文本: {text}")
    
    # 1. 固定种子生成参考音频 (VoiceDesign)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    
    ref_wavs, sr = design_model.generate_voice_design(
        text=text,
        language="Chinese",
        instruct=instruct,
        do_sample=True,
        temperature=0.7,
    )
    
    # 保存参考音频以供检查
    sf.write("ref_style_generated.wav", ref_wavs[0], sr)
    print("✅ 参考音频已生成 (ref_style_generated.wav)")

    # 2. 提取特征构建 Prompt (Base)
    print("🧩 [Setup] 正在提取声纹特征...")
    voice_clone_prompt = base_model.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], sr),
        ref_text=text,
    )
    
    return voice_clone_prompt


# ==================== 3. 播放器类 (含 RTF 统计 & 保存) ====================

class DualModelTTSPlayer:
    def __init__(self, inference_model, clone_prompt, sample_rate=24000):
        self.model = inference_model      # 这里传入 Base 模型
        self.prompt = clone_prompt        # 预计算好的 Prompt
        self.sample_rate = sample_rate
        
        # 队列
        self.text_queue = queue.Queue(maxsize=20)
        self.audio_queue = queue.Queue(maxsize=20)
        
        # 全量音频缓存 (用于保存)
        self.full_audio_buffer = []
        
        # 统计数据
        self.stats = {
            "total_gen_time": 0.0,
            "total_duration": 0.0,
            "chunks": 0
        }

        # Pyaudio 初始化
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        print("🔊 播放器就绪")

    def _generate_worker(self, language="Chinese"):
        """推理线程：Base模型克隆 + RTF统计"""
        print("🎙️ 推理线程启动...")
        
        while True:
            text = self.text_queue.get()
            if text is None: break
                
            self.stats["chunks"] += 1
            print(f"\nProcessing Chunk {self.stats['chunks']}: 「{text[:15]}...」")
            
            # === 计时开始 ===
            t_start = time.time()
            
            try:
                # 固定种子以保证长对话稳定性
                torch.manual_seed(1234) 
                
                # 使用 Base 模型 + Clone Prompt 进行推理
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=self.prompt, # 核心：传入克隆提示词
                    do_sample=True,
                    top_p=0.8,
                    temperature=0.7,
                    repetition_penalty=1.1
                )
                audio_data = wavs[0]
            except Exception as e:
                print(f"❌ 生成错误: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # === 计时结束 ===
            t_end = time.time()
            
            # 计算指标
            gen_time = t_end - t_start
            audio_dur = len(audio_data) / sr
            rtf = gen_time / audio_dur if audio_dur > 0 else 0
            
            # 更新统计
            self.stats["total_gen_time"] += gen_time
            self.stats["total_duration"] += audio_dur
            
            # 存入全量 Buffer
            self.full_audio_buffer.append(audio_data)
            
            print(f"   ⏱️  耗时: {gen_time:.3f}s | 音频: {audio_dur:.2f}s | RTF: {rtf:.3f}")
            if rtf > 1.0:
                print("   ⚠️  [Lag Warning] 生成慢于播放")

            # 放入播放队列
            self.audio_queue.put(audio_data)
        
        self.audio_queue.put(None)
        print("✅ 推理线程结束")

    def _play_worker(self):
        """播放线程"""
        print("🎧 播放线程启动...")
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None: break
            self.stream.write(audio_data.tobytes())
        print("✅ 播放线程结束")

    def start(self, user_input):
        # 启动三个线程：LLM生产 -> TTS消费/生产 -> Player消费
        t_llm = threading.Thread(target=self._stream_llm, args=(user_input,))
        t_gen = threading.Thread(target=self._generate_worker)
        t_play = threading.Thread(target=self._play_worker)
        
        t_play.start()
        t_gen.start()
        t_llm.start()
        
        t_llm.join()
        t_gen.join()
        t_play.join()
        
        self.cleanup()
        self.print_final_stats()

    def _stream_llm(self, user_input):
        """LLM 流式输出 + 分句"""
        buffer = ""
        strong_delimiters = r'[。！？\n]'
        weak_delimiters = r'[，；]'
        
        messages = [{"role": "system", "content": "回答简洁，不带Markdown格式。"}, {"role": "user", "content": user_input}]
        completion = llm_client.chat.completions.create(
            model="qwen-plus-2025-12-01", messages=messages, stream=True, extra_body={"enable_thinking": False}
        )
        
        print("--- LLM Output ---")
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                text = delta.content
                print(text, end="", flush=True)
                buffer += text
                
                # 分句逻辑
                parts = re.split(f'({strong_delimiters})', buffer)
                if len(parts) > 1:
                    for i in range(0, len(parts) - 1, 2):
                        sentence = parts[i] + parts[i+1]
                        if len(sentence.strip()) > 1:
                            self.text_queue.put(sentence.strip())
                    buffer = parts[-1]
                elif len(buffer) > 80:
                    parts = re.split(f'({weak_delimiters})', buffer)
                    if len(parts) > 1:
                        self.text_queue.put((parts[0] + parts[1]).strip())
                        buffer = "".join(parts[2:])
                        
        if buffer.strip():
            self.text_queue.put(buffer.strip())
        self.text_queue.put(None)
        print("\n------------------")

    def save_wav(self, filename="final_output.wav"):
        """保存完整音频"""
        if not self.full_audio_buffer:
            print("⚠️ 无音频数据可保存")
            return
        
        print(f"💾 正在保存全量音频到 {filename} ...")
        full_data = np.concatenate(self.full_audio_buffer)
        sf.write(filename, full_data, self.sample_rate)
        print("✅ 保存成功")

    def print_final_stats(self):
        """打印最终 RTF 报告"""
        total_gen = self.stats["total_gen_time"]
        total_dur = self.stats["total_duration"]
        avg_rtf = total_gen / total_dur if total_dur > 0 else 0
        
        print("\n" + "="*40)
        print(f"📊 性能统计报告")
        print(f"   - 总音频时长 : {total_dur:.2f}s")
        print(f"   - 总推理耗时 : {total_gen:.2f}s")
        print(f"   - 平均 RTF   : {avg_rtf:.3f} (越小越快)")
        if avg_rtf < 1.0:
            print(f"   🚀 状态: 实时 (Real-time)")
        else:
            print(f"   🐢 状态: 非实时 (Non-Real-time)")
        print("="*40 + "\n")

    def cleanup(self):
        if self.stream.is_active(): self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# ==================== 4. 主流程 ====================

if __name__ == "__main__":
    
    # --- 1. 定义想要定制的声音 (只需做一次) ---
    REF_TEXT = "大家好，我是智能助手，很高兴为您服务。"
    # 这里的 instruct 决定了整个对话的音色
    REF_INSTRUCT = "沉稳的央视播音员男声，语速适中，字正腔圆，富有磁性。"
    
    try:
        # --- 2. 预处理：生成 Prompt ---
        # 这一步使用 Design 模型生成音频，并用 Base 模型转为 Prompt
        prompt = create_stable_prompt(REF_TEXT, REF_INSTRUCT, seed=666)
        
        # --- 3. 初始化播放器 (使用 Base 模型 + Prompt) ---
        # 注意：这里传入的是 base_model，后续推理全靠它
        player = DualModelTTSPlayer(base_model, prompt)
        
        # --- 4. 开始对话 ---
        query = "介绍一下北京市。"
        print(f"\n🤖 用户提问: {query}\n")
        
        player.start(query)
        
        # --- 5. 保存结果 ---
        player.save_wav("dialogue_result.wav")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()