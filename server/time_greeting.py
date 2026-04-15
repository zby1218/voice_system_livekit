"""本机本地时间的时段问候与欢迎语，供 Agent 等模块复用。"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

__all__ = ["get_local_now", "time_greeting_of_day", "ruibo_welcome_message"]


def get_local_now() -> datetime:
    """返回本机当前本地日期时间（实时）。"""
    return datetime.now()


def time_greeting_of_day(when: Optional[datetime] = None) -> str:
    """根据本地小时返回时段问候：早上好 / 中午好 / 下午好 / 晚上好。

    划分：05:00–11:59 早上；12:00–13:59 中午；14:00–17:59 下午；其余为晚上。
    """
    dt = when if when is not None else get_local_now()
    h = dt.hour
    if 5 <= h < 12:
        return "早上好"
    if 12 <= h < 14:
        return "中午好"
    if 14 <= h < 18:
        return "下午好"
    return "晚上好"


def ruibo_welcome_message(when: Optional[datetime] = None) -> str:
    """时段问候 + 锐博固定介绍。"""
    return f"{time_greeting_of_day(when)}，我是锐博。"
