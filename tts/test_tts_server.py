#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import queue
import threading

import httpx
import sounddevice as sd
import numpy as np

TTS_URL = "http://127.0.0.1:50000/tts/stream"
TEXT = """

今天，我们带来的，正是这样一款更时尚、更未来的梦想之车——红旗天工“S concept”，接下来，我为大家详细介绍一下红旗天工“S concept”的时尚之魂与未来之境。

先看“更时尚”  红旗天工“S concept”是流动的艺术品，是东方美学与未来科技的完美融合。外观上，整车以“新锐运动轿跑”姿态登场，低趴的车身线条，充满张力与速度感。前脸延续红旗经典的“旗贯长红”设计，并进化为会呼吸的“电子光脉”。标志性的“东方醒狮”大灯，彰显中华文化神韵，凌厉又不失温度。无论是穿行都市，还是高速驰骋，它都自带气场。

进入座舱，外观的时尚气息自然延续到车内。设计灵感源自太空座舱，悬浮的飞船式仪表板，让前沿科技与东方智慧悄然共鸣；星轨式交互扬声器，以科技畅想塑造前沿未来。这不仅是一台车的座舱，更是一个属于年轻人的时尚生活空间。

再看“更未来”  红旗天工“S concept”通过三大技术突破，构筑起这辆未来之车的坚实底座。

一是“大脑”的进化，更聪明，也更懂你。红旗天工“S concept”将搭载红旗独创的舱驾一体方案，打造以AI为原生能力的超级具身智能座舱。自研的中央计算架构让控制器数量减半，由一颗“大脑”集中控制车辆所有功能。同时采用更高效的多模态世界模型，依托强大的算力算法支撑，不仅能听懂指令，更能学习你的习惯、感知你的情绪，在你需要之前，就把车内的一切都安排妥当，让座舱成为一个有温度、能共情、可进化的智能生命体。

二是“身体”的跃迁，更稳，也更安心。红旗天工“S concept”搭载红旗划时代的硬核技术“xx底盘”（当前命名未确定），它是一套具备感知、预判与主动调节能力的“智能四肢”。依托车云协同、多源感知网络以及自研底盘垂直大模型，它不仅看得远，可实时感知前方1000米路况，是人类视觉极限的10倍；还看得准，精准预判200米外起伏、坑洼等路面细节；更开得稳，在你感知到颠簸前，AI底盘已完成200次动态调整，实现“眨眼之间，路况已平”。有了这套AI底盘，新手秒变老司机，每一位乘员都能尽享“无感出行”，全程不颠、不晕、不累。

三是“诞生”的革新，更快交付，也更普惠。红旗首创、国内首个汽车制造“开箱工艺”，将在红旗天工“S concept”量产车型首发。我们打破百年汽车制造流水线模式，将整车解构为六大模块，实现“乐高式”并行装配，结合一体化压铸技术，自动化率大幅提升。以后大家订车，不用再经历漫长等待，提车周期直接缩短一半。全链路成本下降带来的红利，我们将全部让利用户，让顶级智造不再昂贵，让科技真正普惠于民。

更时尚，让我们一眼爱上这台车；更未来，让我们放心把出行交给它。红旗天工“S concept”，将是年轻人表达个性的时尚符号，是都市精英拥抱科技的生活方式，更是中国品牌面向世界的未来宣言。我们相信，这台车，将点燃新一代消费者对美好出行的全部想象，让天工好物，悦美好人生！  

"""


# "一汽红旗以46万年销量、14.9万新能源车的成绩完成华丽转身，全固态电池200℃不起火、产线OTA两小时升级等硬核技术，证明其从央企责任到市场突围的实力。2026年7款新车将携华为全栈方案出击，技术+体系创新正改写中国高端汽车格局。"

def player_worker(sample_rate: int, audio_queue: queue.Queue):
    """独立线程：从队列取音频数据并写入声卡，与 HTTP 接收解耦，避免死锁。

    TTS 流式 chunk 可能为奇数字节，且 httpx 可能在 int16 样本中间截断，必须先拼成偶数字节再 write。
    """
    pending = bytearray()
    with sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    ) as stream:
        while True:
            chunk = audio_queue.get()
            if chunk is None:  # 哨兵值，表示结束
                if len(pending) % 2 == 1:
                    pending.append(0)  # 补齐最后一个样本，避免丢半样本
                if pending:
                    stream.write(pending)
                break
            if chunk:
                pending.extend(chunk)
                n = (len(pending) // 2) * 2
                if n:
                    stream.write(pending[:n])
                    del pending[:n]
            audio_queue.task_done()


def main():
    audio_queue: queue.Queue = queue.Queue()

    # 1) 发起流式请求
    # 本地回环地址测试，不走系统代理，避免 socks:// 代理配置导致 httpx 报错
    with httpx.Client(timeout=None, trust_env=False) as client:
        with client.stream(
            "POST",
            TTS_URL,
            data={
                "text": TEXT,
                "voice_id": "default",
                "session_id": "demo",
                "interrupt_mode": "normal",
            },
        ) as resp:
            resp.raise_for_status()

            # 2) 从响应头拿采样率（服务端会返回 X-Sample-Rate）
            sr = int(resp.headers.get("X-Sample-Rate", "24000"))
            print(f"sample_rate={sr}, start playing...")

            # 3) 启动播放线程（独立线程写声卡，主线程专注接收 HTTP）
            t = threading.Thread(target=player_worker, args=(sr, audio_queue), daemon=True)
            t.start()

            # 4) 主线程持续接收 HTTP 流数据，放入队列，不阻塞
            chunk_count = 0
            for chunk in resp.iter_bytes():
                if chunk:
                    audio_queue.put(chunk)
                    chunk_count += 1
                    print(f"received chunk #{chunk_count}, size={len(chunk)}")

    # 5) 发送哨兵，等待播放线程将队列中所有音频播放完毕后再退出
    audio_queue.put(None)
    t.join()
    print("done.")

if __name__ == "__main__":
    main()