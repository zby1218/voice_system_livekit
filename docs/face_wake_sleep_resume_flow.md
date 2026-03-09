# LiveKit 休眠 → 重启 Face 的流程梳理

按源码走一遍「Agent 判定 away → 发 session_end → Client 发 resume → Face 恢复检测」的完整链路，并标注潜在问题。

---

## 1. 触发休眠（Agent 侧）

**位置**: `livekit-agents/livekit/agents/voice/agent_session.py`

- `user_away_timeout`（默认 15s）内用户与 Agent 都静音时，会调用 `_set_user_away_timer()`（约 1160–1174 行）。
- 定时器到期后执行 `_update_user_state("away")`（约 1218 行）。
- `_update_user_state` 会 `emit("user_state_changed", UserStateChangedEvent(new_state="away"))`（约 1247 行）。

---

## 2. Agent Activity 响应 away

**位置**: `livekit-agents/livekit/agents/voice/agent_activity.py`

- `_on_user_state_changed`（约 855–869 行）在 `event.new_state == "away"` 且 `self.is_awake` 时：
  1. 播 TTS：「我先休息了，有事再叫我吧」
  2. `self.is_awake = False`
  3. `asyncio.create_task(self._notify_client_sleep())`
  4. 清空 KWS 队列（Face 唤醒时 KWS 未用，此处无害）

- `_notify_client_sleep`（约 871–881 行）：
  - 使用 `self._session._room_io.room.local_participant.publish_data(b"session_end", reliable=True)` 向房间发数据。
  - LiveKit 的 `publish_data` 不指定目标时会对房间内其他参与者广播，因此 **Client 会收到**。

---

## 3. Client 收到 session_end 并发 resume

**位置**: `client/client_face.py`

- `@room.on("data_received")` 回调（约 149–156 行）：
  - 若 `data.data == b"session_end"`：
    1. 取 `face_resume_host` / `face_resume_port`（由 `face_wake_listener` 传入，默认 127.0.0.1:9998）
    2. 调用 `_send_resume_to_face(host, port)`：TCP 连到 Face 的 resume 端口，发一行 `{"event":"resume"}\n`
    3. `session_end_event.set()`，让主循环退出

- 主循环（约 204–206 行）：`while not session_end_event.is_set(): await asyncio.sleep(0.5)`，收到 session_end 后退出。
- `finally`（约 208–211 行）：`room.disconnect()`，断开房间。

**结论**: 逻辑正确；Client 在收到休眠信号后先发 resume 再断开，Face 能收到 resume。

---

## 4. Face 的 ResumeServer 收 resume

**位置**: `face/wake_notify.py`

- `ResumeServer` 在**后台线程**中在 `resume_port`（如 9998）上 `listen`，`accept()` 后读一行 UTF-8，解析 JSON。
- 若 `obj.get("event") == "resume"`，则 `self.resume_event.set()`（约 138–139 行）。
- 主线程在 `wait_resume()` 里 `self.resume_event.wait(timeout=...)`，被 set 后返回，并 `resume_event.clear()`（约 109 行），避免下次立即返回。

**结论**: 协议一致（`client_face._send_resume_to_face` 发 `{"event":"resume"}\n`），逻辑正确。

---

## 5. Face 主循环恢复检测

**位置**: `face/face_ratio_detect.py`

- 在 `emit_yes` 且 TCP 通知成功时（约 448–458 行）：
  1. `send_present(...)` 已成功
  2. 若配置了 `resume_server` 和 `resume_camera_factory`：
     - 释放摄像头、关窗口
     - `resume_server.wait_resume()` 阻塞直到收到 resume（或超时）
     - `self.camera = self.resume_camera_factory()` 重新打开摄像头
     - `continue` 回到主循环，继续检测

**结论**: 恢复流程正确，下一轮 present 可再次触发 Client 连接。

---

## 6. face_wake_listener 与端口对应

**位置**: `client/face_wake_listener.py`

- 在 `present_port`（默认 9999）上 `accept()`，收 Face 的 present。
- 解析到 `event == "present"` 后取 `gender`，调用 `run_client(gender=..., face_resume_host=face_host, face_resume_port=face_resume_port)`。
- `face_host` / `face_resume_port` 即 Face 所在主机和 Face 监听的 resume 端口（默认 127.0.0.1:9998）。

**结论**: present 端口（9999）与 resume 端口（9998）、主机配置一致时，整条链路正确。

---

## 流程小结（顺序）

1. **Agent**: user 静音超时 → `user_state_changed(away)` → Activity 播「我先休息了」→ `publish_data(b"session_end")`
2. **Client**: `data_received` → 若 `session_end` → `_send_resume_to_face(host, port)` → `session_end_event.set()` → 主循环退出 → `room.disconnect()`
3. **Face**: ResumeServer 收到 TCP `{"event":"resume"}` → `resume_event.set()` → `wait_resume()` 返回 → 重新打开摄像头 → `continue` 继续检测

---

## 潜在问题与建议

### 1. ~~Client 在回调里阻塞事件循环~~（已修复）

**位置**: `client/client_face.py` 的 `on_data_received`。

- 已改为 `asyncio.create_task(asyncio.to_thread(_send_resume_to_face, host, port))`，在单独线程中发 TCP，不再阻塞事件循环。

### 2. 其它

- **Agent 发 session_end**：未指定 `destination_identities`，房间内所有参与者都会收到；当前只有 Client 一个用户参与者，行为正确。
- **Face 超时**：若长时间未收到 resume（如 Client 崩溃），`wait_resume(timeout=300)` 会超时自动恢复检测，逻辑合理。

---

**结论**: 从「LiveKit 休眠 → 发 session_end → Client 发 TCP resume → Face 恢复」的整条逻辑是**正确且闭环的**。唯一建议是避免在 Client 的 `data_received` 回调里同步发 TCP，改为放到线程或 `to_thread` 中执行。
