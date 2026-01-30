#!/usr/bin/env python3
"""TTS 并发测试脚本

用法:
    python test_concurrent.py --requests 5 --concurrent 3
"""

import asyncio
import time
import argparse
import httpx
from typing import List, Dict, Any


async def send_tts_request(
    client: httpx.AsyncClient,
    request_id: int,
    text: str,
    url: str = "http://localhost:50000/inference_sft"
) -> Dict[str, Any]:
    """发送单个 TTS 请求并测量时间"""
    
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0
    
    print(f"📤 [请求 #{request_id}] 发送: '{text[:30]}...'")
    
    try:
        async with client.stream("POST", url, data={"tts_text": text, "spk_id": "default"}, timeout=120) as response:
            if response.status_code != 200:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                }
            
            async for chunk in response.aiter_bytes():
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    ttfb = first_chunk_time - start_time
                    print(f"⚡ [请求 #{request_id}] 首包延迟: {ttfb:.3f}s")
                
                total_bytes += len(chunk)
                chunk_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ [请求 #{request_id}] 完成 | 总时间: {total_time:.3f}s | 数据: {total_bytes/1024:.1f}KB | 块数: {chunk_count}")
        
        return {
            "request_id": request_id,
            "success": True,
            "start_time": start_time,
            "first_chunk_time": first_chunk_time,
            "end_time": end_time,
            "ttfb": first_chunk_time - start_time if first_chunk_time else None,
            "total_time": total_time,
            "total_bytes": total_bytes,
            "chunk_count": chunk_count,
        }
    
    except Exception as e:
        print(f"❌ [请求 #{request_id}] 失败: {e}")
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
        }


async def check_health(url: str = "http://localhost:50000/health") -> Dict[str, Any]:
    """检查服务器健康状态"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5)
        return response.json()


async def run_concurrent_test(
    num_requests: int = 5,
    max_concurrent: int = 3,
    base_url: str = "http://localhost:50000"
) -> None:
    """运行并发测试"""
    
    # 检查服务器状态
    print("=" * 60)
    print("🔍 检查服务器状态...")
    try:
        health = await check_health(f"{base_url}/health")
        print(f"   状态: {health.get('status')}")
        print(f"   GPU: {health.get('gpu')}")
        print(f"   最大并发: {health.get('max_concurrent')}")
        print(f"   可用音色: {health.get('voices')}")
    except Exception as e:
        print(f"❌ 无法连接服务器: {e}")
        return
    
    print("=" * 60)
    print(f"🚀 开始并发测试: {num_requests} 个请求, 最多 {max_concurrent} 个同时发送")
    print("=" * 60)
    
    # 准备测试文本（不同长度）
    test_texts = [
        "你好，这是第一个测试请求，我们来测试一下并发性能。",
        "第二个测试请求，希望能够看到多个请求同时处理的效果。",
        "这是第三个测试，文本稍微长一点，看看处理时间会不会增加很多。",
        "第四个请求来了，让我们继续测试并发能力。",
        "最后一个测试请求，完成后我们来看看总体的性能数据。",
        "额外的测试请求一，用于更多的测试场景。",
        "额外的测试请求二，继续增加测试样本。",
        "额外的测试请求三，最后的测试数据。",
    ]
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_request(request_id: int, text: str, client: httpx.AsyncClient):
        async with semaphore:
            return await send_tts_request(client, request_id, text, f"{base_url}/inference_sft")
    
    # 开始计时
    total_start = time.time()
    
    async with httpx.AsyncClient() as client:
        # 创建所有任务
        tasks = [
            limited_request(i + 1, test_texts[i % len(test_texts)], client)
            for i in range(num_requests)
        ]
        
        # 并发执行
        results = await asyncio.gather(*tasks)
    
    total_end = time.time()
    total_elapsed = total_end - total_start
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 测试结果统计")
    print("=" * 60)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"   总请求数: {num_requests}")
    print(f"   成功: {len(successful)}")
    print(f"   失败: {len(failed)}")
    print(f"   总耗时: {total_elapsed:.3f}s")
    
    if successful:
        ttfbs = [r["ttfb"] for r in successful if r.get("ttfb")]
        total_times = [r["total_time"] for r in successful]
        
        print(f"\n   首包延迟 (TTFB):")
        print(f"      最小: {min(ttfbs):.3f}s")
        print(f"      最大: {max(ttfbs):.3f}s")
        print(f"      平均: {sum(ttfbs)/len(ttfbs):.3f}s")
        
        print(f"\n   单请求总时间:")
        print(f"      最小: {min(total_times):.3f}s")
        print(f"      最大: {max(total_times):.3f}s")
        print(f"      平均: {sum(total_times)/len(total_times):.3f}s")
        
        # 计算并发效率
        sequential_time = sum(total_times)
        efficiency = sequential_time / total_elapsed
        print(f"\n   并发效率:")
        print(f"      串行总时间 (估计): {sequential_time:.3f}s")
        print(f"      实际总时间: {total_elapsed:.3f}s")
        print(f"      加速比: {efficiency:.2f}x")
    
    if failed:
        print(f"\n   失败详情:")
        for r in failed:
            print(f"      请求 #{r['request_id']}: {r.get('error')}")
    
    print("=" * 60)


async def monitor_health(base_url: str = "http://localhost:50000", interval: float = 1.0):
    """持续监控服务器状态"""
    print("🔍 开始监控服务器状态 (Ctrl+C 停止)")
    print("-" * 50)
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(f"{base_url}/health", timeout=2)
                data = response.json()
                active = data.get("active_requests", 0)
                max_c = data.get("max_concurrent", 0)
                total = data.get("total_requests", 0)
                
                bar = "█" * active + "░" * (max_c - active)
                print(f"\r[{bar}] 活跃: {active}/{max_c} | 累计: {total}", end="", flush=True)
            except Exception as e:
                print(f"\r❌ 连接失败: {e}", end="", flush=True)
            
            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="TTS 并发测试")
    parser.add_argument("--requests", type=int, default=5, help="总请求数")
    parser.add_argument("--concurrent", type=int, default=3, help="最大同时发送数")
    parser.add_argument("--url", type=str, default="http://localhost:50000", help="TTS 服务器地址")
    parser.add_argument("--monitor", action="store_true", help="持续监控模式")
    
    args = parser.parse_args()
    
    if args.monitor:
        try:
            asyncio.run(monitor_health(args.url))
        except KeyboardInterrupt:
            print("\n\n🛑 监控已停止")
    else:
        asyncio.run(run_concurrent_test(
            num_requests=args.requests,
            max_concurrent=args.concurrent,
            base_url=args.url
        ))


if __name__ == "__main__":
    main()
