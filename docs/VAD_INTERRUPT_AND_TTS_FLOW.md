# VAD 打断后 LLM/TTS 逻辑与 TTS 请求堆积说明

本文档沿源码说明：VAD 触发打断时 LLM 与 TTS 的处理、给 TTS 的请求会怎样，以及为何在频繁打断时可能出现 TTS 请求堆积、系统无声音。

---

## 1. VAD 何时触发“打断”

- **入口**：`AgentActivity` 实现 `RecognitionHooks`，VAD 结果通过 `AudioRecognition` 回调到 activity。
- **触发打断的回调**：
  - **`on_vad_inference_done`**（`agent_activity.py` 约 1368–1375 行）：当 `ev.speech_duration >= min_interruption_duration` 时调用 `_interrupt_by_audio_activity()`。
  - **`on_interim_transcript`**（约 1385–1404 行）：有临时识别结果时也会调用 `_interrupt_by_audio_activity()`。
  - **`on_final_transcript`**（约 1414–1447 行）：最终识别结果时再次调用 `_interrupt_by_audio_activity()`，保证即使用户没被 VAD 判成“正在说话”，也能在 STT 出结果时打断。

即：**只要 VAD 或 STT 认为“用户有在说话/有结果”，当前可打断的播报就会被标记为打断**。

---

## 2. 打断时对“当前话轮”做了什么（LLM 侧不主动停）

- **`_interrupt_by_audio_activity()`**（约 1300–1335 行）：
  - 若有 `_current_speech` 且 `allow_interruptions=True` 且尚未被标记为打断：
    - 可选：若支持 pause，则只暂停播放（`output.audio.pause()`）；
    - 否则：`_current_speech.interrupt()`，且若有 Realtime 会话则 `_rt_session.interrupt()`。
  - **这里只做两件事**：把当前 `SpeechHandle` 标记为“已打断”，以及通知 Realtime 会话。**并不会在这里取消 LLM 或 TTS 的 asyncio 任务**。

- **`SpeechHandle.interrupt()` → `_cancel()`**（`speech_handle.py` 约 117–197 行）：
  - 将 `_interrupt_fut.set_result(None)`，即“打断”状态生效。
  - **不立刻 cancel 任何 task**；只是启动一个 **5 秒定时器**（`INTERRUPTION_TIMEOUT`）：若 5 秒后该 speech 仍未 `done()`，才 `task.cancel()` 所有 `_tasks` 并 `_mark_done()`。
  - 因此：**真正让 LLM/TTS 停下来的，不是 interrupt() 本身，而是下面要说的“谁在 await 这些任务”对 `_interrupt_fut` 的响应以及后续的 cancel**。

结论：**LLM 在生成 chunk 时若被打断，不会在“LLM 内部”被停掉；是否停止由“跑 LLM+TTS 的那层逻辑”通过 `wait_if_not_interrupted` 和后续 `cancel_and_wait` 决定。**

---

## 3. 谁在“跑”LLM 和 TTS，以及打断后是否还给 TTS 发数据

- **一条完整回复（LLM→TTS）** 在 **`_pipeline_reply_task_impl`**（`agent_activity.py` 约 1924–2125 行）里跑：
  - `perform_llm_inference()` 得到 `llm_task` 和 `llm_gen_data`；
  - `llm_gen_data.text_ch` 经 `tee` 分成两路：一路给 **TTS**（`tts_text_input`），一路给转录等；
  - `perform_tts_inference(node, input=tts_text_input, ...)` 创建 **tts_task**，TTS 从 `tts_text_input`（即 LLM 的 `text_ch`）读 **流式 chunk**；
  - 这些任务都放进 `tasks`，最后用 **`speech_handle.wait_if_not_interrupted([*tasks])`** 一起等（约 2104 行）。

- **`wait_if_not_interrupted`**（`speech_handle.py` 约 169–174 行）：
  - `await asyncio.wait([gather(tasks), self._interrupt_fut], return_when=FIRST_COMPLETED)`。
  - 一旦 **`_interrupt_fut` 完成**（即上面某处调用了 `interrupt()`），这里 **立即返回**，不再等 LLM/TTS 自然结束。

- **返回之后在 `_pipeline_reply_task_impl` 里**（约 2112–2114、2131–2134 行）：
  - 若 `speech_handle.interrupted`：
    - 会 **`await utils.aio.cancel_and_wait(*tasks)`**，即 **取消并等待 llm_task、tts_task、forward_task 等**。
  - 因此：**打断后，当前这条 pipeline 的 LLM 和 TTS 任务会被 cancel，之后 LLM 不会再往 `text_ch` 里写，TTS 也不会再从这个 pipeline 向 TTS 服务发新请求。**

- **“后续回答还给 TTS 吗？”**  
  - **被中断的这条回复**：打断后就不会再给 TTS 发新内容（任务被 cancel，channel 会随任务结束/关闭而不再有数据）。
  - **下一次用户话轮**：会由 `on_end_of_turn` → `_user_turn_completed_task` 里 **重新** `_generate_reply()`，产生 **全新的** `SpeechHandle` 和全新的 LLM+TTS pipeline，**新的回答会正常走 TTS**。所以：**“后续回答”指的是“下一轮回复”，会正常给 TTS；当前轮被打断的那条，不会再把后续 chunk 给 TTS。**

---

## 4. 给 TTS 的请求在打断时会发生什么

- **数据流**：LLM 的 chunk → `text_ch` → tee → `tts_text_input` → `perform_tts_inference` → 你项目里的 **CosyVoiceTTS**（`tts/custom_tts.py`）。
- **CosyVoice 行为**（`CosyVoiceChunkedStream._run()`）：
  - 按段（segment）循环，每段一次 **HTTP POST**（`_build_request_for_segment` + `self._tts._client.stream(...)`）。
  - 没有在循环里检查“当前 SpeechHandle 是否已 interrupt”或 asyncio 取消；**只要上层还在消费这个 stream，就会一直发段**。
- **上层何时“不再消费”**：
  - 当 `_pipeline_reply_task_impl` 里执行 **`cancel_and_wait(*tasks)`** 时，**tts_task** 被 cancel。
  - `tts_task` 里是在 `generation.py` 的 `_tts_inference_task` 中 **async for** 消费 TTS 的 stream；cancel 会向该 async generator 抛 **CancelledError**，从而结束对 TTS stream 的消费。
- **因此**：
  - **尚未发出去的段**：cancel 后就不会再发。
  - **已经发出去的 HTTP 请求**：请求已经到 TTS 服务端，**不会被客户端主动 abort**（除非底层 HTTP 客户端在 cancel 时主动关连接）；服务端会照常处理完这些请求，结果可能不再被播放（因为上层已 cancel，不再收音频）。

所以：**“给 TTS 的请求”在打断后 = 已发出的请求会继续在服务端执行，但不再有新请求从这条 pipeline 发出；未发出的段不会再发。**

---

## 5. 为何“频繁 VAD 打断”时容易 TTS 请求堆积、系统无声音

可能原因可以归纳为几类：

1. **同一时刻多条 pipeline 在往 TTS 发请求**
   - 用户快速连续说话/打断时：上一轮的 `_user_turn_completed_task` 可能还没跑完（例如还在 `await old_task` 或刚创建新 pipeline），下一轮 **on_end_of_turn** 又触发，再起一条新 pipeline。
   - 代码在 1674–1678 行：若“已有更新的 user turn”会 **interrupt 刚创建的 speech_handle**，所以会存在“刚创建就被打断”的 pipeline；但在被打断前，这条 pipeline 可能已经 **开始了 LLM 推理并给 TTS 发了第一段（甚至多段）**。
   - 于是：**多条 pipeline（有的已打断、有的新开）在短时间内都向 TTS 服务发请求**，TTS 服务端队列里请求变多。

2. **打断后已发出的请求仍占满 TTS 服务**
   - CosyVoice 是“按段”发 HTTP，每段一个请求。打断后客户端不再发新段，但 **已发出去的请求** 仍在服务端排队/执行。
   - 若 TTS 服务是单线程或并发有限，会先处理完这些“已废弃对话”的请求，**新对话的请求排在后面**，听感上就是“没声音”或延迟很大。

3. **scheduling 与“谁在播”的时序**
   - `_scheduling_task` 按队列依次 `_authorize_generation()` 播报；若当前 speech 被 interrupt，会 `speech._mark_done()` 等，然后播下一个。
   - 若多个 speech 被快速打断，队列里会有一串“已打断”的 handle，它们对应的 TTS 请求可能仍在服务端执行，**占住资源**，导致“该播的新回复”的 TTS 请求迟迟得不到处理，听不到声音。

4. **5 秒 INTERRUPTION_TIMEOUT 的兜底**
   - 若因某种原因 `cancel_and_wait(*tasks)` 没有很快结束（例如 TTS 卡在某个 HTTP 调用），要等 **5 秒** 后 `SpeechHandle._cancel()` 的定时器才会强制 cancel 所有 `_tasks`。
   - 在这 5 秒内，该 pipeline 理论上可能还在占着 TTS 连接或仍有未结束的请求，加剧堆积。

---

## 5.1 补充：TTS 首帧很快（如 0.25s）时，堆积从哪来？

若 TTS 服务端单请求首帧延迟很低（例如 0.25s），“旧请求太慢堵住新请求”的说法不够准确。更符合现象的是下面两种叠加：

1. **后续请求（同一 pipeline 的 2、3、4… 段）在打断前已经发出**
   - 首段：LLM 出一段 → TTS 发请求 1 → 0.25s 首帧播出，正常。
   - 之后 LLM 继续流式出字，TTS 按段切分并**连续发请求 2、3、4…**。
   - 用户在这段时间内打断：`interrupt()` 到 `wait_if_not_interrupted` 返回、再 `cancel_and_wait(tts_task)` 有一小段延迟；在这段延迟内，**该 pipeline 可能已经又发出了多段 HTTP 请求**。
   - 这些**已发出的后续请求**既不会撤回，服务端也会照常排队、推理、返回；只是播放端可能因打断已清 buffer，用户听不到。

2. **多条 pipeline 并发，每条都贡献 1～多段**
   - 快速连续打断时，会多次触发 `on_end_of_turn`，每条新话轮起一条新 `_generate_reply()`（新 pipeline）。
   - 每条 pipeline 在被打断前都可能已经：发了首段（0.25s 首帧），并继续发 2、3 段…
   - 于是：**请求堆积 = 多 pipeline × 每 pipeline 在 cancel 前已发出的多段**，而不是“单请求太慢”。

**结果**：服务端推理单次可能很快，但**请求数量**在频繁打断时涨得很快（多 pipeline + 每 pipeline 的后续段），队列变长；新对话的 TTS 请求排在这些“已废弃”的请求后面，听感上就是延迟大或像没声音。也就是说：**反常主要来自“后续请求”在打断前已经发出 + 多 pipeline 并发，导致请求量堆积，而不是单次推理慢。**

| 问题 | 结论 |
|------|------|
| VAD 打断后 LLM 还继续生成吗？ | 会继续生成直到 **wait_if_not_interrupted 返回后** 执行 **cancel_and_wait(*tasks)**；之后 LLM 任务被 cancel，不再产生新 chunk。 |
| 打断后还会给 TTS 发新请求吗？ | **当前这条 pipeline** 不会：tts_task 被 cancel，不再从 text_ch 读，CosyVoice 不再发新段。**下一轮新回复** 会走新的 pipeline，会正常给 TTS。 |
| 已发给 TTS 的请求呢？ | 已发出的 HTTP 请求不会被客户端主动撤销，会在 TTS 服务端照常执行，可能加剧服务端队列堆积。 |
| 为何频繁打断时 TTS 堆积、无声音？ | 多轮 pipeline 在短时间内都向 TTS 发请求；打断后旧请求仍在服务端执行；新回复的请求被排在后面，表现成无声音或严重延迟。 |

---

## 7. 可考虑的改进方向（实现时需再对源码）

- **TTS 客户端**：在 CosyVoiceChunkedStream 的循环中检查 `speech_handle.interrupted` 或 asyncio 当前任务是否被取消，一旦打断则 **停止发新段** 并尽量 **结束/关闭当前 HTTP stream**（若底层 HTTP 支持 abort）。
- **服务端**：TTS 服务端对“同一会话/同一 turn_id”的请求做优先级或废弃策略（例如只执行最新一条，取消或丢弃旧请求），减少无效计算和队列堆积。
- **Agent 侧**：在 `cancel_and_wait(*tasks)` 之后，若 TTS 暴露“关闭/取消当前合成”的 API，可主动调一次，避免已无用的请求继续占用服务端。

以上逻辑均基于当前仓库中 `livekit-agents` 的 `agent_activity.py`、`generation.py`、`speech_handle.py` 以及 `tts/custom_tts.py` 的阅读，若你改过相关逻辑，需要再对照实际代码做一次核对。

---

## 8. 在 tts_server.py 里记录什么可以判断“堆积”

不改 agent 代码的前提下，在 **tts_server.py** 侧建议至少记录下面几类数据，用来确认是否存在“请求排队/堆积”以及是否和频繁打断相关。

### 8.1 建议必记（判断是否在排队）

| 记录项 | 含义 | 记录时机 | 用来判断什么 |
|--------|------|----------|----------------|
| **queue_wait_s** | 本请求从“想拿槽位”到“拿到槽位”的等待时间（秒） | 在 `_acquire_slot()` 内：`acquire()` 之前打点 `t_before`，`acquire()` 返回后打点 `t_after`，`queue_wait_s = t_after - t_before` | 若经常 > 0 且有时较大，说明有请求在等槽位，存在排队/堆积 |
| **active_count_at_acquire** | 拿到槽位那一刻，正在处理的请求数（含本请求） | `_acquire_slot()` 里在 `_active_count += 1` 之后立刻记一次 | 若经常等于 `max_concurrent`，说明槽位常满，新请求只能等 |
| **request_id** | 请求序号 | 现有逻辑已有 | 和上面两项一起写日志，便于按请求分析 |

**结论**：若在“频繁打断”场景下，日志里 **queue_wait_s** 经常明显 > 0（例如 > 0.1s 甚至数秒），且 **active_count_at_acquire** 经常等于 max_concurrent，就可以判断：**堆积发生在 TTS 服务端排队（等槽位）**，和前面说的“多 pipeline / 后续段在打断前已发出”一致。

### 8.2 建议选记（区分“排队”和“单次推理慢”）

| 记录项 | 含义 | 记录时机 | 用来判断什么 |
|--------|------|----------|----------------|
| **ttfb_s** | 从拿到槽位到第一次 yield 出首包的时间（秒） | 在 `synthesize_zero_shot` 的 generator 里，第一次 `yield` 之前：用“当前时间 - 拿槽位时间” | 若 ttfb_s 稳定在 ~0.25s 左右，而 queue_wait_s 很大，说明延迟主要来自**排队**，不是单次推理慢 |
| **total_s** | 从拿到槽位到 `_release_slot()` 的时间 | 在 `_release_slot()` 里：用“当前时间 - 该 request_id 的拿槽位时间” | 看单次请求总占用槽位时长，和请求速率一起可估队列深度 |

若 **ttfb_s 一直很小、queue_wait_s 在打断多时变大**，就能确认：服务端单次推理很快，延迟/无声音主要来自**请求量多、槽位满、新请求排队**。

### 8.3 记录格式建议

便于后续筛“频繁打断”时段，建议每条请求至少打一行结构化日志，例如：

- 拿槽位时：`request_id`, `queue_wait_s`, `active_count_at_acquire`, `ts`（可选）
- 首包时（若记）：`request_id`, `ttfb_s`, `ts`
- 释放时（若记）：`request_id`, `total_s`, `ts`

可先只实现 **8.1**，用 **queue_wait_s** 和 **active_count_at_acquire** 就能判断“是否因堆积导致延迟/无声音”；再根据需要加 8.2 做更细分析。

---

## 9. 根据 log 判断 TTS 模型推理是否有问题

当前 tts_server 会打 `[TTS-Metrics]` 和 Slot/首包等日志，可这样判断：

| 现象 | 日志表现 | 结论 |
|------|----------|------|
| **推理本身慢** | `ttfb_s` 经常很大（如 > 1s、数秒） | 单次推理慢，可能是模型/GPU/文本长度导致，需看模型或资源 |
| **排队导致延迟** | `queue_wait_s` 经常 > 0（如 > 0.1s），且 `ttfb_s` 正常（如 ~0.25s） | 延迟主要来自等槽位，不是推理慢；多为多 pipeline/打断导致请求堆积 |
| **槽位常满** | `active_count_at_acquire` 经常等于 `max_concurrent` | 并发打满，新请求只能排队 |
| **单次请求占用过长** | `total_s` 很大（如 > 10s） | 单段合成时间过长，可能是文本过长或模型/IO 问题 |
| **无异常** | `queue_wait_s` ≈ 0，`ttfb_s` 稳定在 ~0.25s，`total_s` 合理 | 推理正常，无明显堆积或推理问题 |

**结论**：若日志里 **ttfb_s 小且稳定**，而 **queue_wait_s** 在频繁打断时变大，说明 **TTS 模型推理没问题**，问题在请求排队；若 **ttfb_s 经常很大**，才需要排查模型/GPU/配置。

---

## 10. 为何 TTS 不排队后仍会听到「第一句一段 + 第二句一小段 + 第三句一小段 + 第四句完整」

在 TTS 服务端做了「打断时取消同会话旧请求」后，服务端不再排队，但 agent 端仍可能出现：**第一问播了一小段、第二问一小段、第三问一小段、第四问完整**。原因在 **agent 侧播放链路的时序**，与 **clear_buffer 未清 RTC 轨道队列**。

### 10.1 时序：每轮在打断前已把「首段」送进播放

- 每次用户说完一句（Q1/Q2/Q3/Q4），会触发 `on_end_of_turn` → `_user_turn_completed_task`。
- 该任务会：先 interrupt 当前 speech、再创建新 speech 并 `_schedule_speech` 入队；**调度是串行的**，同一时刻只有一个 speech 在跑 pipeline（LLM+TTS+forward）。
- 但对**单条 pipeline** 而言：TTS 按段发请求，**首段返回很快**（如 0.25s），首段音频会立刻通过 `audio_output.capture_frame()` 进入 **Room 的音频输出**：
  1. 先进入 `_ParticipantAudioOutput` 的 `_audio_buf`（channel）；
  2. `_forward_audio` 从 channel 取帧，再送入 `_audio_source.capture_frame()`，即 **LiveKit 的 RTC 轨道队列**（`queue_size_ms=200`）。
- 用户若在**首段已经进入轨道队列甚至已开始播放**之后才触发「下一句」的打断，则：
  - 当前 pipeline 被 interrupt → 后续会执行 `clear_buffer()` + `wait_for_playout()`；
  - 但 **已经进入 RTC 轨道队列的那一段音频** 不会被「撤回」，会照常播完。
- 因此每一轮（Q1、Q2、Q3）都可能在打断前把 **当前句的首段** 已经推进轨道，于是听感是：**第一句一段、第二句一小段、第三句一小段**；最后一句（Q4）没再被打断，所以 **第四句完整**。

### 10.2 为何 clear_buffer 没有清掉「已进轨道」的音频

- **`clear_buffer()`**（`room_io/_output.py` 的 `_ParticipantAudioOutput`）当前只做两件事：
  1. `_audio_bstream.clear()`：清掉尚未压成帧的字节流；
  2. `_interrupted_event.set()`：让 `_forward_audio` 之后收到的帧不再往 `_audio_source` 送。
- **没有** 调用 **`_audio_source.clear_queue()`**，即 **RTC 轨道内部队列** 不会被清空。
- 而 **`_audio_source.clear_queue()`** 只在 **`_wait_for_playout()`** 里、在「检测到 interrupted」分支里被调用；`_wait_for_playout()` 又只有在 **`flush()`** 被调用时才会以 `_flush_task` 形式跑起来。
- 若在**首段尚未 flush**（即还没触发「当前段结束」的 flush）时就打断，则不会启动 `_wait_for_playout()`，**轨道队列里的首段就不会被清掉**，会照常播完。

所以：**打断时若只调用 `clear_buffer()` 而不清 RTC 轨道队列，已进入轨道的首段（甚至多段）仍会按顺序播完**，表现为「多句各播一小段」。

### 10.3 修复思路（与实现对应）

1. **在 `clear_buffer()` 里同时清空 RTC 轨道队列**  
   在 `_ParticipantAudioOutput.clear_buffer()` 中增加 **`_audio_source.clear_queue()`**，这样一旦打断，**已送入轨道但尚未播出的音频** 也会被丢弃，不会再把「上一句的首段」播完。

2. **在打断时立刻清一次音频输出**  
   在 **`_user_turn_completed_task`** 里，在对当前 speech 和队列里的 speech 执行 **interrupt 之后**，立刻对 **`self._session.output.audio`** 调用 **`clear_buffer()`**（若存在且可清）。这样不依赖「当前 pipeline 稍后执行到 clear_buffer」的时序，**一打断就清**，避免在 pipeline 还没反应过来时又有一两帧被推进轨道。

---

## 11. 为何在 10.3 修改后仍会「先播问题1一句 → 问题2一句 → 问题3一句 → 正常播问题4」（仅分析）

在已做「clear_buffer 时清 RTC 轨道队列」以及「打断时立即 clear 音频输出」后，仍出现：**先播放问题1回答的一句，再跳到问题2的一句、问题3的一句，然后正常播放问题4**。下面是仅基于源码的原因分析（不改代码）。

### 11.1 数据流与「谁在清空」

- **写入路径**：TTS → `generation._audio_forwarding_task` 的 `async for frame in tts_output` → `audio_output.capture_frame(frame)` → `_ParticipantAudioOutput.capture_frame()` → `_audio_bstream.push()` → **`_audio_buf.send(f)`**（`room_io/_output.py` 约 104、122 行）。即所有帧都先进入 **`_audio_buf`**（`utils.aio.Chan`）。
- **读出路径**：只有 **`_forward_audio`** 在消费：`async for frame in self._audio_buf`（约 193 行）→ 若未打断则 `_audio_source.capture_frame(frame)`（约 215 行）。即帧从 `_audio_buf` 被取走后，才进入 **`_audio_source`**（RTC 轨道，`queue_size_ms=200`）。
- **clear_buffer() 当前做的**（约 135–143 行）：  
  `_audio_bstream.clear()`、`_interrupted_event.set()`、`_audio_source.clear_queue()`。  
  **没有**对 **`_audio_buf`** 做任何 drain/清空。
- **`_audio_buf` 何时被清空**：  
  - 路径一：**`_forward_audio`** 在 `_interrupted_event.is_set()` 时对取到的帧直接 `continue` 丢弃，相当于一边取一边丢，直到 channel 里没数据（或等 `_flush_task`）。  
  - 路径二：**`_wait_for_playout()`** 在「interrupted」分支里（约 177–178 行）`while not self._audio_buf.empty(): self._audio_buf.recv_nowait().duration`，把 channel 里剩余帧抽空。  
  `_wait_for_playout()` 只有在 **`flush()`** 被调用时才会以 `_flush_task` 形式跑起来；而 **`flush()` 只在 `generation._audio_forwarding_task` 的 `finally` 里调用**（generation 约 410 行），即要等 **forward task 被 cancel 并退出** 后才会执行。因此存在时序：activity 已经执行了 interrupt + clear_buffer()，但 **cancel_and_wait(*tasks)** 尚未完成，forward task 还没退出，**flush() 还没被调用**，`_flush_task` 尚未创建。这段时间里只有 `_forward_audio` 在「取一帧丢一帧」地消耗 `_audio_buf`；若 producer（forward task）已不再送新帧，最终 `_audio_buf` 会被丢光。所以单从「channel 里剩不剩旧帧」看，只要有一次 flush() 执行并跑完 `_wait_for_playout()`，interrupted 分支会 drain `_audio_buf`，且 `_interrupted_event` 会在 185 行被 clear，下一句不会从 channel 里误播旧帧。  
  **结论**：在现有逻辑下，**更可能的问题不是「_audio_buf 里旧帧没清干净、被下一句播出来」**，而是下面 11.2 的「已进轨道 / 已播放」部分。

### 11.2 根本原因：已进 RTC 轨道且已播放（或已提交设备）的音频无法撤回

- **clear_queue() 能清掉什么**：只清 **`_audio_source`（RTC 轨道）内部队列里、尚未被 track 取走送到底层/设备** 的缓冲。
- **clear_queue() 清不掉的**：**已经被 track 从队列里取出并提交给设备播放** 的那一段。一旦帧被取走，就无法再「撤回」。
- **时序**：每个 pipeline 的 TTS 是按「段」返回的（CosyVoice 按段 HTTP），**一段往往对应一整句**。首段首包很快（如 0.25s），agent 会在很短时间内把这一段的多帧依次：  
  `capture_frame` → `_audio_buf.send` → `_forward_audio` 取出 → `_audio_source.capture_frame(frame)`。  
  即**在用户来得及打断之前，第一句（第一个 segment）的多帧可能已经全部或大部分进入 `_audio_source`，且其中一部分已被 track 取走并开始播放**。此时再执行 clear_buffer()（含 clear_queue()），只能清掉**还在轨道队列里、未播放**的后续帧；**已经播放或已提交给设备的那一句无法被清掉**，所以会听到「问题1的一句」。
- 同理：问题2、问题3 各自的首段在被打断前也有机会把「一句」送进轨道并开始播放，clear_queue() 只能清后续，已播的那一句仍会听到。问题4 没再被打断，所以会完整播完。

### 11.3 为何是「一句」而不是「一小段」

- TTS 端按「段」返回，**一段** 通常对应一个 HTTP 请求的整块音频，在 CosyVoice 里往往按句或按标点切，所以**一个 segment ≈ 一句**。
- 首段返回快，在「用户说完并触发打断」之前，该段的多帧已经大量进入轨道并开始播放，因此听感是**每个问题各「一句」**，而不是「几毫秒的一小段」。

### 11.4 小结（仅分析）

| 层级 | 是否在打断时被清空 | 说明 |
|------|--------------------|------|
| **`_audio_bstream`** | 是 | clear_buffer() 里直接 clear。 |
| **`_audio_source` 队列** | 是 | clear_buffer() 里 clear_queue()，但只清「未播放」部分。 |
| **已送入 track 且已播放/已提交设备** | 否 | 无法撤回，会照常播完，表现为「每个问题各一句」。 |
| **`_audio_buf`（Channel）** | 间接 | 未在 clear_buffer() 里显式 drain；靠 _forward_audio 丢弃或 _wait_for_playout() 的 interrupted 分支 recv_nowait 抽空；在 flush() 被调用并跑完 _wait_for_playout() 后，不会留下旧帧给下一句播。 |

**根本原因**：**RTC 轨道层只能清「尚未播放」的队列；已经播放或已提交给设备的那一段无法清掉。** 加上 TTS 首段往往是一整句、且首段很快就被推满轨道并开始播放，所以打断发生时「当前这一句」往往已经播出一部分或即将播完，clear_queue() 只能清后面，听感上就是「问题1一句 → 问题2一句 → 问题3一句 → 问题4完整」。

---

## 12. 从 LLM 首 token 到 LiveKit 播放的完整调用链（仅梳理，不改代码）

以下按「LLM 发出第一个 token → 作为 chunk 交给 TTS → TTS 生成音频 → 交给 LiveKit 播放」的顺序，用与 11 节相同的细粒度方式写出全流程调用与数据结构。

---

### 12.1 入口与 pipeline 的建立

- **入口**：一次用户话轮结束后，`AgentActivity._user_turn_completed_task`（`agent_activity.py` 约 1529 行）里会创建本条回复的 pipeline；若走「LLM + TTS」分支，会进入 **`_pipeline_reply_task_impl`**（约 1935 行）。
- **创建 LLM 与 TTS 的 channel**：
  - **`perform_llm_inference(...)`**（`generation.py` 约 58–81 行）：创建 **`text_ch = aio.Chan[Union[str, FlushSentinel]]()`**，并启动 **`_llm_inference_task`**；`llm_task` 在 done 时会 **`text_ch.close()`**（约 71 行）。
  - **`text_tee = utils.aio.itertools.tee(llm_gen_data.text_ch, 2)`**（`agent_activity.py` 约 1988 行）：得到两路消费，**`tts_text_input`** 为其中一路，供 TTS 消费；另一路 `tr_input` 供转录等使用。
  - **`perform_tts_inference(node=tts_node, input=tts_text_input, ...)`**（约 2002–2008 行）：内部创建 **`audio_ch = aio.Chan[rtc.AudioFrame]()`**（`generation.py` 约 204 行），并启动 **`_tts_inference_task`**；TTS task 在 done 时会 **`audio_ch.close()`**（约 216 行）。
- **创建音频转发**：**`perform_audio_forwarding(audio_output=..., tts_output=tts_gen_data.audio_ch)`**（`agent_activity.py` 约 2083–2086 行）会启动 **`_audio_forwarding_task`**，该 task 从 **`tts_output`**（即 `audio_ch`）读帧并写入 **`audio_output`**（即 `self._session.output.audio`，Room 下为 **`_ParticipantAudioOutput`**）。

因此：**LLM 输出 → `text_ch` → tee 出一路 `tts_text_input` → TTS 消费并写入 `audio_ch` → 转发 task 从 `audio_ch` 读到 `audio_output`**。

---

### 12.2 LLM 发出第一个 token，到写入 text_ch（chunk 的由来）

- **`_llm_inference_task`**（`generation.py` 约 86–186 行）：
  - **`llm_node = node(chat_ctx, tools, model_settings)`**（约 114 行）：这里 **`node`** 即 **`self._agent.llm_node`**；默认实现在 **`agent.py`** 约 398–419 行：**`activity_llm.chat(...)`** 得到 **`stream`**，**`async for chunk in stream`** 从底层 LLM API（如 OpenAI）拉流式 chunk。
  - 对**第一个 content chunk**：若 `isinstance(chunk, str)`（约 138 行），则 **`text_ch.send_nowait(chunk)`**（约 141 行），并 **`data.first_chunk_fut.set_result(None)`**（约 141–142 行）；若 `isinstance(chunk, ChatChunk)` 且 **`chunk.delta.content`** 非空（约 164–167 行），则 **`text_ch.send_nowait(chunk.delta.content)`**（约 166 行），并同样 **`first_chunk_fut.set_result(None)`**。
  - **FlushSentinel**：LLM 若 yield **`FlushSentinel`**（约 169–170 行），会 **`text_ch.send_nowait(chunk)`**，用于在 TTS 侧标记「当前段结束」。

因此：**底层 LLM 流式接口 → `async for chunk in stream` → 第一个 content 出现时 `text_ch.send_nowait(chunk)`，且 `first_chunk_fut` 被 set → 后续 chunk 持续写入 `text_ch`**。  
**LiveKit 侧「等待其为 chunk」** 体现在：**`await llm_gen_data.first_chunk_fut`**（`agent_activity.py` 约 1996 行）在真正发出第一个 chunk 之后才打 E2E 打点并启动 TTS 侧逻辑。

---

### 12.3 从 text_ch 到 TTS 的「输入流」：tee 与 _tts_inference_task

- **TTS 消费的输入**：**`tts_text_input`** 是 **`tee(llm_gen_data.text_ch, 2)`** 的一路（`agent_activity.py` 约 1988–1989 行），因此从 **同一个 `text_ch`** 里读出的仍是 **`str | FlushSentinel`**。
- **`_tts_inference_task`**（`generation.py` 约 224–306 行）：
  - 用 **`itertools.tee(input, 2)`** 再拆成两路（约 273 行）：一路给 **`_get_start_time`**（约 278 行）用于打 TTFB 的「首 token 到达时间」；另一路给 **`_input_segment()`**（约 283–287 行），**`async for chunk in input_tee[1]`**，遇到 **`FlushSentinel`** 则 return，否则 **`yield chunk`**（字符串）。
  - **`_tts_node_inference(input_segment, pushed_duration)`**（约 235–271 行）：**`tts_node = node(input, model_settings)`**（约 239 行），这里 **`node`** 即 **`self._agent.tts_node`**，**`input`** 即为上面 **`_input_segment()`** 的 async generator，即「按段 yield 字符串、遇 FlushSentinel 段结束」的流。
  - 在 **`async for audio_frame in tts_node`**（约 255 行）中，每收到一帧就 **`audio_ch.send_nowait(audio_frame)`**（约 269 行）；首帧时还会在约 256–257 行记录 **`data.ttfb`**。

因此：**`text_ch` 中的 chunk（str/FlushSentinel）→ tee 出 `tts_text_input` → `_tts_inference_task` 的 `input` → `_input_segment()` 按段 yield 字符串 → 作为 `tts_node(agent, text, model_settings)` 的 `text` 参数传入**。

---

### 12.4 TTS 节点：从文本流到 audio_frame 流（含 CosyVoice + StreamAdapter）

- **默认 `Agent.tts_node`**（`agent.py` 约 422–450 行）：
  - 若 TTS 无 streaming 能力（如 CosyVoice 的 **`streaming=False`**），会用 **`tts.StreamAdapter`** 包装（约 431–435 行）。
  - **`wrapped_tts.stream(conn_options=conn_options)`** 得到 **`stream`**（约 438 行），即 **`StreamAdapterWrapper`**（`stream_adapter.py` 约 78–147 行）。
  - **`_forward_input()`**（约 440–444 行）：**`async for chunk in text`** 从上游拿到字符串，对每个 chunk 调用 **`stream.push_text(chunk)`**，最后 **`stream.end_input()`**。
  - **`async for ev in stream`**（约 448 行）：从 **`stream`** 取到 **`ev`**（带 **`ev.frame`** 的合成事件），**`yield ev.frame`**（约 449 行）还给 **`_tts_inference_task`**。

- **StreamAdapterWrapper._run**（`stream_adapter.py` 约 89–146 行）：
  - **`sent_stream = self._tts._sentence_tokenizer.stream()`**（约 90 行）：按句切分的流。
  - **`_forward_input()`**（约 109–116 行）：**`async for data in self._input_ch`**（上游 push_text 写入的 token / FlushSentinel），非 FlushSentinel 则 **`sent_stream.push_text(data)`**，最后 **`sent_stream.end_input()`**。
  - **`_synthesize()`**（约 118–136 行）：**`async for ev in sent_stream`**，每得到一个句子 **`ev.token`**，就 **`async with self._tts._wrapped_tts.synthesize(text, ...) as tts_stream`**（约 130 行）；对 CosyVoice，**`synthesize`** 返回 **`CosyVoiceChunkedStream`**（`custom_tts.py` 约 206–207 行）。**`async for audio in tts_stream`**（约 133 行）得到 **`audio.frame`**，**`output_emitter.push(audio.frame.data.tobytes())`**（约 134 行），段末 **`output_emitter.flush()`**（约 136 行）。这些 push/flush 最终会变成 **SynthesizeStream** 对外 yield 的、带 **`.frame`** 的合成事件。

- **CosyVoiceChunkedStream**（`custom_tts.py` 约 213–463 行）：
  - **`_run`** 内按 **`_split_dialogue(self.input_text)`** 切段（约 361 行），对每段 **`seg`** 调用 **`_build_request_for_segment(seg)`** 得到 url/data/files，用 **`self._tts._client.stream("POST", candidate, data=data, files=files)`**（约 386 行）发 HTTP 流式请求；**`resp.aiter_bytes()`** 取 body，首包用 **`asyncio.wait_for(it.__anext__(), ...)`** 等首帧（约 405–408 行），然后 **`output_emitter.push(first)`**（约 415 行），后续 **`output_emitter.push(chunk)`**（约 433 行），段间可 **`output_emitter.push(self._silence_bytes(...))`**（约 449 行），最后 **`output_emitter.flush()`**（约 451 行）。**output_emitter** 由框架按 PCM 转成 **`rtc.AudioFrame`** 并注入到 stream 的 event 中，因此上层 **`async for audio in tts_stream`** 拿到的 **`audio.frame`** 即 **`rtc.AudioFrame`**。

因此：**上游 text 流（str + FlushSentinel）→ StreamAdapterWrapper.push_text / end_input → 按句 tokenizer → 每句 synthesize → CosyVoiceChunkedStream 按段 HTTP 流式请求 → output_emitter.push(bytes) / flush → 框架封装为带 .frame 的 event → tts_node 的 async for ev in stream 得到 ev.frame → _tts_inference_task 里 audio_ch.send_nowait(audio_frame)**。

---

### 12.5 从 audio_ch 到 Room 的 audio_output：转发与 capture_frame

- **`perform_audio_forwarding(audio_output, tts_output=tts_gen_data.audio_ch)`**（`generation.py` 约 349–356 行）：创建 **`_audio_forwarding_task`**，**`tts_output`** 即为 **`audio_ch`**。
- **`_audio_forwarding_task`**（约 361–411 行）：
  - **`async for frame in tts_output`**（约 369 行）：从 **`audio_ch`** 取 **`rtc.AudioFrame`**。
  - 若需重采样会先 **`resampler.push(frame)`**，否则直接 **`await audio_output.capture_frame(frame)`**（约 390 行）；若重采样则对 **`resampler.push(frame)`** 产生的帧逐个 **`await audio_output.capture_frame(f)`**（约 389–390 行）。
  - 首帧后 **`out.first_frame_fut.set_result(None)`**（约 395 行）。
  - 循环结束后 **`audio_output.flush()`**（约 410 行），用于标记「当前段结束」并触发 Room 侧 **`_flush_task`**（见下）。

因此：**`audio_ch` 中的 `rtc.AudioFrame` → _audio_forwarding_task 的 async for frame in tts_output → 可选 resampler → audio_output.capture_frame(frame) → 段结束 audio_output.flush()**。  
这里的 **`audio_output`** 即 **`self._session.output.audio`**，在 Room 场景下为 **`_ParticipantAudioOutput`**（`room_io/_output.py`）。

---

### 12.6 Room 侧：capture_frame → _audio_buf → _forward_audio → RTC 轨道

- **`_ParticipantAudioOutput.capture_frame(frame)`**（`room_io/_output.py` 约 94–116 行）：
  - **`await super().capture_frame(frame)`**（约 97 行）：基类会更新 playback 段计数等。
  - **`for f in self._audio_bstream.push(frame.data)`**（约 103–105 行）：**`_audio_bstream`** 是 **`utils.audio.AudioByteStream`**，按固定块切分；每个切出的 **`f`** 是 **`rtc.AudioFrame`**，**`await self._audio_buf.send(f)`**（约 104 行）写入 **`_audio_buf`**（**`utils.aio.Chan[rtc.AudioFrame]`**，约 45 行），并累加 **`_pushed_duration`**；首帧时记录 **`_first_frame_time`**（约 106–107 行）。

- **`_forward_audio`**（约 192–221 行）：常驻任务，**`async for frame in self._audio_buf`**（约 193 行）从 **`_audio_buf`** 取帧；若 **`_interrupted_event.is_set()` 或 _pushed_duration == 0**（约 208 行）则 **continue**（不送 track）；否则 **`await self._audio_source.capture_frame(frame)`**（约 215 行）。**`_audio_source`** 是 **`rtc.AudioSource(sample_rate, num_channels, queue_size_ms=200)`**（约 39 行），即 LiveKit 的 RTC 轨道输入；其内部队列由 **`clear_queue()`** 清空（见 11 节）。

- **`flush()`**（约 118–133 行）：**`_audio_bstream.flush()`** 产生的尾帧会 **`self._audio_buf.send_nowait(f)`**（约 122 行）；若有 **`_pushed_duration`** 则创建 **`self._flush_task = asyncio.create_task(self._wait_for_playout())`**（约 133 行），用于等待本段播完或被打断，并在 **`_wait_for_playout()`** 里在 interrupted 时 **drain _audio_buf**（recv_nowait）并 **clear_queue()**（约 177–182 行）。

因此：**audio_output.capture_frame(frame) → _audio_bstream.push(frame.data) → _audio_buf.send(f) → _forward_audio 的 async for frame in _audio_buf → 未打断则 _audio_source.capture_frame(frame) → RTC 轨道排队/播放**；**flush() 时尾帧再入 _audio_buf，并启动 _wait_for_playout()**。

---

### 12.7 全流程串联（简要）

| 阶段 | 位置 | 行为与数据结构 |
|------|------|----------------|
| 1. LLM 首 token | `generation._llm_inference_task`，`agent.llm_node` | 底层 LLM stream → `async for chunk in llm_node` → 首个 content 时 **`text_ch.send_nowait(chunk)`**，**`first_chunk_fut.set_result(None)`**。 |
| 2. LiveKit 等首 chunk | `agent_activity._pipeline_reply_task_impl` | **`await llm_gen_data.first_chunk_fut`** 后再打 E2E 打点并建 TTS。 |
| 3. chunk 到 TTS 输入 | `generation` tee + `_tts_inference_task` | **`text_tee = tee(llm_gen_data.text_ch, 2)`**，**`tts_text_input`** → **`_tts_inference_task`** 的 **`input`** → **`_input_segment()`** 按段 **yield str**，遇 **FlushSentinel** 段结束。 |
| 4. TTS 消费文本、产音频 | `agent.tts_node`，StreamAdapter，CosyVoice | **stream.push_text(chunk)** 写入句 tokenizer；按句 **synthesize(text)** → **CosyVoiceChunkedStream** 按段 HTTP 流 → **output_emitter.push(bytes)** / **flush** → 封装为 **ev.frame**；**async for ev in stream** → **yield ev.frame**。 |
| 5. 音频写入 audio_ch | `generation._tts_inference_task` | **`async for audio_frame in tts_node`** → **`audio_ch.send_nowait(audio_frame)`**。 |
| 6. 转发到 output | `generation._audio_forwarding_task` | **`async for frame in tts_output`**（即 **audio_ch**）→ 可选 resampler → **`await audio_output.capture_frame(frame)`**；段末 **`audio_output.flush()`**。 |
| 7. Room 写入 channel | `room_io._output._ParticipantAudioOutput.capture_frame` | **`_audio_bstream.push(frame.data)`** → 每个 **f** 做 **`await self._audio_buf.send(f)`**。 |
| 8. 从 channel 到 RTC 轨道 | `room_io._output._forward_audio` | **`async for frame in self._audio_buf`** → 若未 **`_interrupted_event.is_set()`** 则 **`await self._audio_source.capture_frame(frame)`**；**flush()** 时再送尾帧并启动 **`_wait_for_playout()`**。 |

以上即为从 **LLM 发出第一个 token**，到 **作为 chunk 交给 TTS**，再到 **TTS 生成音频**，最后到 **LiveKit 播放** 的完整调用与数据流；未修改任何代码，仅做流程梳理。
