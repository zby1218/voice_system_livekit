"""
人脸唤醒策略模块：仅封装 Face 唤醒逻辑（kws_enabled=False 时使用）。
KWS 唤醒代码保持在 agent_activity 中不动，不在此模块内实现。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IFaceWakeActivity(Protocol):
    """Activity 提供给 FaceWakeHandler 的接口。"""

    _session: Any

    def say(self, text: str, *, allow_interruptions: bool = True) -> None: ...


class FaceWakeHandler:
    """人脸唤醒：视为已唤醒，会话启动时播唤醒语并更新状态。"""

    async def on_session_start(self, activity: IFaceWakeActivity) -> None:
        activity.say("我在呢", allow_interruptions=False)
        activity._session._update_user_state("listening")
