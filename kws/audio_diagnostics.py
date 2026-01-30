#!/usr/bin/env python3
"""
音频诊断工具 - 用于分析和可视化 KWS 音频数据
用法:
    1. 在 kws_server.py 中启用诊断模式，录制音频
    2. 运行此脚本分析录制的音频文件
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import json
import wave

# 尝试导入 scipy 用于频谱分析
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found, spectrogram will be disabled")


def configure_chinese_font():
    """
    自动检测并配置 matplotlib 的中文字体
    """
    import matplotlib
    from matplotlib.font_manager import FontManager
    import platform
    
    system = platform.system()
    
    # 常用中文字体列表 (按优先级排序)
    # Windows: SimHei (黑体), Microsoft YaHei (微软雅黑)
    # Linux: WenQuanYi Micro Hei (文泉驿微米黑), Noto Sans CJK SC
    # Mac: Heiti TC (黑体), PingFang SC
    font_candidates = [
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', # Linux 优先
        'SimHei', 'Microsoft YaHei', 'SimSun',                          # Windows
        'Heiti TC', 'PingFang SC', 'Arial Unicode MS'                   # Mac
    ]
    
    # 1. 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 2. 尝试找到一个可用的中文字体
    fm = FontManager()
    found_font = None
    available_fonts = {f.name for f in fm.ttflist}
    
    for font_name in font_candidates:
        if font_name in available_fonts:
            found_font = font_name
            break
            
    # 3. 应用字体
    if found_font:
        plt.rcParams['font.sans-serif'] = [found_font] + plt.rcParams['font.sans-serif']
        # print(f"✅ Matplotlib 使用中文字体: {found_font}")
    else:
        # Linux 如果没有字体，回退方案
        if system == 'Linux':
            print("⚠️ 未检测到常用中文字体，图表中文可能乱码。建议安装: sudo apt install fonts-wqy-microhei")
        else:
            print("⚠️ 未检测到常用中文字体，请检查系统字体设置")

# 执行配置
configure_chinese_font()

@dataclass
class AudioStats:
    """音频统计信息"""
    source: str                    # 数据来源 (kws_test / livekit_agent)
    duration_s: float              # 录制时长 (秒)
    sample_rate: int               # 采样率
    total_samples: int             # 总采样数
    
    # 振幅统计
    amplitude_min: float           # 最小振幅
    amplitude_max: float           # 最大振幅
    amplitude_mean: float          # 平均振幅
    amplitude_std: float           # 振幅标准差
    
    # 能量统计
    rms_energy: float              # RMS 能量
    peak_db: float                 # 峰值 (dB)
    rms_db: float                  # RMS (dB)
    
    # 信号质量
    zero_crossing_rate: float      # 零交叉率
    dynamic_range_db: float        # 动态范围 (dB)
    
    # 数据格式
    is_normalized: bool            # 是否已归一化 [-1, 1]
    data_range: str                # 数据范围描述
    
    # 时间戳
    recorded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def print_report(self):
        """打印诊断报告"""
        print("\n" + "=" * 60)
        print(f"📊 音频诊断报告 - {self.source}")
        print("=" * 60)
        print(f"📅 录制时间: {self.recorded_at}")
        print(f"⏱️  时长: {self.duration_s:.2f}s ({self.total_samples} samples)")
        print(f"🎵 采样率: {self.sample_rate} Hz")
        print("-" * 60)
        print("【振幅统计】")
        print(f"   范围: [{self.amplitude_min:.4f}, {self.amplitude_max:.4f}]")
        print(f"   均值: {self.amplitude_mean:.6f}")
        print(f"   标准差: {self.amplitude_std:.4f}")
        print(f"   数据格式: {self.data_range}")
        print(f"   已归一化: {'是' if self.is_normalized else '否'}")
        print("-" * 60)
        print("【能量统计】")
        print(f"   RMS 能量: {self.rms_energy:.6f}")
        print(f"   峰值: {self.peak_db:.1f} dB")
        print(f"   RMS: {self.rms_db:.1f} dB")
        print(f"   动态范围: {self.dynamic_range_db:.1f} dB")
        print("-" * 60)
        print("【信号质量】")
        print(f"   零交叉率: {self.zero_crossing_rate:.4f}")
        print("=" * 60)


class AudioDiagnostics:
    """音频诊断工具"""
    
    def __init__(self, sample_rate: int = 16000, output_dir: str = "./diagnostics"):
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 录制缓冲区
        self._recording = False
        self._buffer: list[bytes] = []
        self._start_time: Optional[float] = None
        self._source: str = "unknown"
    
    def start_recording(self, source: str = "unknown"):
        """开始录制"""
        self._recording = True
        self._buffer = []
        self._start_time = time.time()
        self._source = source
        print(f"🔴 开始录制音频 (来源: {source})...")
    
    def push_audio(self, audio_bytes: bytes):
        """推送音频数据"""
        if self._recording:
            self._buffer.append(audio_bytes)
    
    def stop_recording(self) -> Optional[str]:
        """停止录制并保存"""
        if not self._recording:
            return None
        
        self._recording = False
        duration = time.time() - self._start_time if self._start_time else 0
        
        if not self._buffer:
            print("⚠️ 没有录制到数据")
            return None
        
        # 合并数据
        combined = b''.join(self._buffer)
        audio_data = np.frombuffer(combined, dtype=np.int16)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._source}_{timestamp}"
        
        # 保存原始数据
        raw_path = os.path.join(self.output_dir, f"{filename}.npy")
        np.save(raw_path, audio_data)
        print(f"💾 原始数据已保存: {raw_path}")
        
        # 保存 WAV 文件
        wav_path = os.path.join(self.output_dir, f"{filename}.wav")
        self._save_wav(wav_path, audio_data)
        print(f"💾 WAV 音频已保存: {wav_path}")
        
        # 分析并生成报告
        stats = self.analyze(audio_data, source=self._source, duration_hint=duration)
        stats.print_report()
        
        # 保存统计信息
        stats_path = os.path.join(self.output_dir, f"{filename}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(stats.to_json())
        
        # 生成可视化
        plot_path = os.path.join(self.output_dir, f"{filename}.png")
        self.plot(audio_data, stats, save_path=plot_path)
        print(f"📈 可视化已保存: {plot_path}")
        
        return filename
    
    def analyze(self, audio_data: np.ndarray, source: str = "unknown", 
                duration_hint: Optional[float] = None) -> AudioStats:
        """分析音频数据"""
        # 基本信息
        total_samples = len(audio_data)
        duration = duration_hint if duration_hint else total_samples / self.sample_rate
        
        # 转为 float 进行分析
        audio_float = audio_data.astype(np.float32)
        
        # 判断数据格式
        abs_max = np.max(np.abs(audio_float))
        if abs_max <= 1.0:
            is_normalized = True
            data_range = "[-1.0, 1.0] (归一化)"
        elif abs_max <= 32768:
            is_normalized = False
            data_range = "[-32768, 32767] (int16)"
            # 归一化用于后续分析
            audio_float = audio_float / 32768.0
        else:
            is_normalized = False
            data_range = f"[{audio_float.min():.0f}, {audio_float.max():.0f}] (异常)"
        
        # 振幅统计
        amp_min = float(np.min(audio_float))
        amp_max = float(np.max(audio_float))
        amp_mean = float(np.mean(audio_float))
        amp_std = float(np.std(audio_float))
        
        # 能量统计
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        peak = float(np.max(np.abs(audio_float)))
        
        # dB 计算 (避免 log(0))
        peak_db = 20 * np.log10(peak + 1e-10)
        rms_db = 20 * np.log10(rms + 1e-10)
        dynamic_range = peak_db - rms_db
        
        # 零交叉率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_float))) > 0)
        zcr = zero_crossings / len(audio_float)
        
        return AudioStats(
            source=source,
            duration_s=duration,
            sample_rate=self.sample_rate,
            total_samples=total_samples,
            amplitude_min=amp_min,
            amplitude_max=amp_max,
            amplitude_mean=amp_mean,
            amplitude_std=amp_std,
            rms_energy=rms,
            peak_db=peak_db,
            rms_db=rms_db,
            zero_crossing_rate=zcr,
            dynamic_range_db=dynamic_range,
            is_normalized=is_normalized,
            data_range=data_range,
        )
    
    def plot(self, audio_data: np.ndarray, stats: AudioStats, 
             save_path: Optional[str] = None, show: bool = False):
        """生成可视化图表"""
        # 归一化数据用于绘图
        audio_float = audio_data.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0
        
        # 时间轴
        time_axis = np.arange(len(audio_float)) / self.sample_rate
        
        # 创建图表
        fig, axes = plt.subplots(3 if HAS_SCIPY else 2, 1, figsize=(14, 10))
        fig.suptitle(f"音频诊断 - {stats.source}\n{stats.recorded_at}", fontsize=14)
        
        # 1. 波形图
        ax1 = axes[0]
        ax1.plot(time_axis, audio_float, linewidth=0.5, color='steelblue')
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel("振幅")
        ax1.set_title(f"波形图 | RMS: {stats.rms_db:.1f}dB | Peak: {stats.peak_db:.1f}dB")
        ax1.set_xlim(0, time_axis[-1])
        ax1.set_ylim(-1.1, 1.1)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. 振幅包络 (RMS 窗口)
        ax2 = axes[1]
        window_size = int(0.02 * self.sample_rate)  # 20ms 窗口
        if window_size > 0:
            rms_envelope = np.array([
                np.sqrt(np.mean(audio_float[i:i+window_size]**2))
                for i in range(0, len(audio_float) - window_size, window_size // 2)
            ])
            env_time = np.linspace(0, time_axis[-1], len(rms_envelope))
            ax2.plot(env_time, rms_envelope, color='coral', linewidth=1)
            ax2.fill_between(env_time, 0, rms_envelope, alpha=0.3, color='coral')
        ax2.set_xlabel("时间 (s)")
        ax2.set_ylabel("RMS 能量")
        ax2.set_title(f"能量包络 (20ms 窗口) | 零交叉率: {stats.zero_crossing_rate:.4f}")
        ax2.set_xlim(0, time_axis[-1])
        ax2.grid(True, alpha=0.3)
        
        # 3. 频谱图 (如果 scipy 可用)
        if HAS_SCIPY:
            ax3 = axes[2]
            f, t, Sxx = signal.spectrogram(audio_float, self.sample_rate, 
                                           nperseg=256, noverlap=128)
            # 转为 dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            im = ax3.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
            ax3.set_xlabel("时间 (s)")
            ax3.set_ylabel("频率 (Hz)")
            ax3.set_title("频谱图")
            ax3.set_ylim(0, 8000)  # 显示到 8kHz
            plt.colorbar(im, ax=ax3, label='功率 (dB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 图表已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _save_wav(self, path: str, audio_data: np.ndarray):
        """保存为 16bit PCM WAV 文件"""
        # 确保数据是 int16
        if audio_data.dtype != np.int16:
             # 如果是 float，尝试转换
             if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                 # 假设 float 是 [-1, 1]
                 if np.max(np.abs(audio_data)) <= 1.0:
                     audio_data = (audio_data * 32767).astype(np.int16)
                 else:
                     audio_data = audio_data.astype(np.int16)
             else:
                 audio_data = audio_data.astype(np.int16)

        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

def compare_reports(report1_path: str, report2_path: str):
    """对比两个诊断报告"""
    with open(report1_path, 'r') as f:
        stats1 = json.load(f)
    with open(report2_path, 'r') as f:
        stats2 = json.load(f)
    
    print("\n" + "=" * 70)
    print("📊 音频对比报告")
    print("=" * 70)
    print(f"{'指标':<25} {'来源1':>20} {'来源2':>20}")
    print("-" * 70)
    
    keys_to_compare = [
        ('source', '来源'),
        ('duration_s', '时长 (s)'),
        ('sample_rate', '采样率'),
        ('amplitude_min', '振幅最小值'),
        ('amplitude_max', '振幅最大值'),
        ('rms_energy', 'RMS 能量'),
        ('peak_db', '峰值 (dB)'),
        ('rms_db', 'RMS (dB)'),
        ('zero_crossing_rate', '零交叉率'),
        ('is_normalized', '已归一化'),
    ]
    
    for key, label in keys_to_compare:
        v1 = stats1.get(key, 'N/A')
        v2 = stats2.get(key, 'N/A')
        if isinstance(v1, float):
            v1 = f"{v1:.4f}"
        if isinstance(v2, float):
            v2 = f"{v2:.4f}"
        print(f"{label:<25} {str(v1):>20} {str(v2):>20}")
    
    print("=" * 70)


# 全局诊断实例 (供 kws_server 使用)
_diagnostics: Optional[AudioDiagnostics] = None

def get_diagnostics(output_dir: str = "./diagnostics") -> AudioDiagnostics:
    """获取全局诊断实例"""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = AudioDiagnostics(output_dir=output_dir)
    return _diagnostics


if __name__ == "__main__":
    # 命令行模式: 分析已保存的 .npy 文件
    if len(sys.argv) < 2:
        print("用法:")
        print("  分析单个文件: python audio_diagnostics.py <file.npy>")
        print("  对比两个报告: python audio_diagnostics.py compare <stats1.json> <stats2.json>")
        sys.exit(0)
    
    if sys.argv[1] == "compare" and len(sys.argv) == 4:
        compare_reports(sys.argv[2], sys.argv[3])
    else:
        # 分析 npy 文件
        npy_path = sys.argv[1]
        if not os.path.exists(npy_path):
            print(f"文件不存在: {npy_path}")
            sys.exit(1)
        
        audio_data = np.load(npy_path)
        diag = AudioDiagnostics()
        source = os.path.basename(npy_path).replace('.npy', '')
        stats = diag.analyze(audio_data, source=source)
        stats.print_report()
        
        # 保存统计信息
        stats_path = npy_path.replace('.npy', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(stats.to_json())
        print(f"💾 统计信息已保存: {stats_path}")
        
        # 保存 WAV 文件
        wav_path = npy_path.replace('.npy', '.wav')
        diag._save_wav(wav_path, audio_data)
        print(f"💾 WAV 音频已保存: {wav_path}")
        
        # 生成可视化
        plot_path = npy_path.replace('.npy', '.png')
        diag.plot(audio_data, stats, save_path=plot_path, show=True)
