# livekit-agents 本地修改与 pip 版对比（解耦说明）

## 一、agent_session.py 差异

| 项目 | pip 版 (site-packages) | 本地 (livekit-agents/) |
|------|------------------------|------------------------|
| **AgentSessionOptions** | 含 `aec_warmup_duration`，无 `kws_enabled` | 含 `kws_enabled`，无 `aec_warmup_duration` |
| **AgentSession.__init__** | 无 `kws_enabled` 参数；有 `aec_warmup_duration=3.0` | 有 `kws_enabled=False` 参数 |
| **Recording** | `record: bool \| RecordingOptions`，细粒度（audio/traces/logs/transcript） | `record: bool`，`_enable_recording` |
| **其他** | 有 `ClientEventsHandler`、`MetricsReport`、`AgentTask`、`run(..., input_modality)` 等新 API | 基于旧版 + 仅增加 KWS 开关 |

**解耦要点**：pip 版没有 `kws_enabled`，在 `stt_llm_agent.py` 里**不要**传 `kws_enabled`。唤醒与打断完全由本项目的「常驻 KWS 监听」负责。

---

## 二、agent_activity.py 差异

| 项目 | pip 版 | 本地修改 |
|------|--------|----------|
| **唤醒/休眠** | 无 `is_awake`，音频始终进 STT/VAD | `is_awake` + `_face_wake`（人脸唤醒时直接置 True）；未唤醒时音频只进 `_kws_queue` |
| **内置 KWS** | 无 | `_kws_queue`、`_kws_task`、`_run_kws()`，连 ws://localhost:8765，收到 `wake_detected` 置 `is_awake=True`、清队列、`_update_user_state("listening")` |
| **人脸唤醒** | 无 | `FaceWakeHandler`，`kws_enabled=False` 时 `on_session_start` 将 `is_awake=True` |
| **user_state_changed** | 无自定义 | 监听 `user_state_changed`，`away` 时播「我先休息了…」、`is_awake=False`、`_notify_client_sleep()` 发 `session_end` |
| **push_audio** | 有 `aec_warmup_remaining` 的 discard 逻辑 | 无 AEC warmup；先判断 `is_awake`，未唤醒则只 `_kws_queue.put_nowait` 并 return |
| **interrupt** | 无 `clear_buffer()` | 打断时调用 `session.output.audio.clear_buffer()` |
| **_interrupt_by_audio_activity** | 有 AEC warmup 期间不打断 | 无 AEC；打断时同样 `clear_buffer()`；打 pipeline 日志 |
| **E2ETimingEvent** | 无 | 在首 token 时 `emit("e2e_timing", E2ETimingEvent(llm_first_token=...))` |
| **commit_user_turn** | 有 `skip_reply` 参数，会 `generate_reply()` | 无 `skip_reply`，签名不同 |
| **_schedule_speech** | 无 TIMING 日志；delay 计算略有不同 | 有 `[TIMING]` 日志；`_actual_delay` 计算与日志 |
| **say / generate_reply** | 有 `input_details`（InputDetails） | 无 `input_details` |
| **其他** | `_cancel_speech_pause_task`、`_stt_eos_received`、`AgentConfigUpdate` 等 | `_interrupt_paused_speech_task`；部分逻辑简化 |

**解耦要点**：

- pip 版**没有**「未唤醒只送 KWS」的逻辑，音频会一直进 STT/VAD，因此用 pip 时**不需要**在 session 里区分 kws/face，由本项目在 `stt_llm_agent.py` 的常驻 KWS 做唤醒 + 打断 + `say("我在听")` 即可。
- pip 版**没有** `E2ETimingEvent`，`stt_llm_agent.py` 不要订阅 `e2e_timing`，用 `metrics_collected` 里的 `LLMMetrics.ttft` 即可（你已有）。
- pip 版**有** `clear_user_turn()`、`interrupt(force=True)`、`session.say()`，常驻 KWS 回调里可以照常调用。

---

## 三、stt_llm_agent.py 使用 pip 包时的修改清单

1. **不再传 `kws_enabled`**  
   `AgentSession(...)` 去掉参数 `kws_enabled=kws_enabled`（pip 的 `AgentSession` 无此参数）。

2. **不再订阅 `e2e_timing`**  
   去掉 `session.on("e2e_timing", _on_e2e_timing)`（pip 不发出该事件；LLM 首包耗时已用 `_on_metrics_collected` 的 `LLMMetrics.ttft`）。

3. **（可选）人脸唤醒**  
   pip 版没有内置「人脸唤醒即 is_awake」逻辑。若仍需人脸唤醒后立刻播一句，可在 `session.start(...)` 之后根据 `attributes.get("wake_source") == "face"` 调一次 `session.say("我在听", allow_interruptions=False)`，由你决定是否保留。

4. **保留**  
   - 常驻 KWS：`_run_always_on_kws_listener`、`track_subscribed` 里创建任务、`clear_user_turn()`、`activity.interrupt(force=True)`、`session.say("我在听", allow_interruptions=False)`。  
   - `allow_interruptions=False`。  
   - 从 `livekit.agents.voice.room_io` 导入 `RoomOptions`、`AudioInputOptions`（pip 也有）。  
   - `playback_boundary_log` 的 try/except 保留（pip 无此模块会走 except）。

按上述修改后，可完全使用 pip 安装的 `livekit-agents` 与 `livekit-plugins-*`，不再依赖本地 `livekit-agents/` 源码。
