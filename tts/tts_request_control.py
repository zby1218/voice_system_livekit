"""
TTS 会话级请求控制模块。

目标：
- 支持按会话(session_id)取消旧请求（而不是全局取消）。
- 支持 queued/running 请求统一设置取消标记。
- 降低 tts_server.py 中的状态管理耦合。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

InterruptMode = Literal["normal", "latest_only"]


@dataclass(frozen=True)
class RequestTicket:
    request_id: int
    session_id: str
    event: threading.Event


class SessionRequestController:
    """会话级请求控制器（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_request_id = 0
        self._request_events: Dict[int, threading.Event] = {}
        self._request_session: Dict[int, str] = {}
        self._session_requests: Dict[str, Set[int]] = {}

    def create_request(
        self, session_id: str, interrupt_mode: InterruptMode
    ) -> Tuple[RequestTicket, List[int]]:
        """创建请求票据，并按策略取消同会话旧请求。返回(新票据, 被取消请求ID列表)。"""
        cancelled: List[int] = []
        with self._lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            ev = threading.Event()
            self._request_events[request_id] = ev
            self._request_session[request_id] = session_id
            self._session_requests.setdefault(session_id, set()).add(request_id)

            if interrupt_mode == "latest_only":
                for rid in sorted(self._session_requests.get(session_id, set())):
                    if rid == request_id:
                        continue
                    event = self._request_events.get(rid)
                    if event is not None and not event.is_set():
                        event.set()
                        cancelled.append(rid)

        return RequestTicket(request_id=request_id, session_id=session_id, event=ev), cancelled

    def is_cancelled(self, request_id: int) -> bool:
        with self._lock:
            ev = self._request_events.get(request_id)
            return bool(ev and ev.is_set())

    def cancel_all(self) -> List[int]:
        """取消所有请求（保留给管理接口，生产中慎用）。"""
        cancelled: List[int] = []
        with self._lock:
            for rid, ev in self._request_events.items():
                if not ev.is_set():
                    ev.set()
                    cancelled.append(rid)
        return cancelled

    def remove_request(self, request_id: int) -> Optional[str]:
        """请求结束时清理状态，返回其 session_id（若存在）。"""
        with self._lock:
            session_id = self._request_session.pop(request_id, None)
            self._request_events.pop(request_id, None)
            if session_id is None:
                return None
            s = self._session_requests.get(session_id)
            if s is not None:
                s.discard(request_id)
                if not s:
                    self._session_requests.pop(session_id, None)
            return session_id
