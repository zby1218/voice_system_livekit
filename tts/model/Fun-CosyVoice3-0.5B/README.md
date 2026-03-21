# Fun-CosyVoice3-0.5B（占位目录）

本目录在仓库中**仅保留路径**，`.pt` / `.onnx` 等模型文件**不会**随 Git 提交。

部署时请自行将 **CosyVoice3 0.5B** 完整模型文件放到本目录（与 `tts_server.py` 默认 `--model-dir` 一致），例如：

- 从官方 / Hugging Face / 内部分发渠道下载与训练导出一致的目录结构；
- 或从已有机器打包拷贝整个 `Fun-CosyVoice3-0.5B` 文件夹覆盖到此处。

安装完成后，本目录下应包含 `cosyvoice3.yaml`、`llm.pt`、`flow.pt` 等运行所需文件（以你使用的发行包为准）。
