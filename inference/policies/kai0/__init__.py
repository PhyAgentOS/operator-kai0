"""kai0 策略适配器实现。

- adapter.py：WS 同步模式（Kai0SyncAdapter，短连接 + chunk 摊销）；
- rtc_adapter.py：RTC 异步推理模式（Kai0RtcAdapter，常驻连接 + 后台推理线程）；
- stream_buffer.py：RTC 动作流缓冲（trim/blend/snapshot，两模式共用组件之一）；
- payloads.py：观测组包 / 响应拆包共享逻辑（同步与 RTC adapter 共用）。
"""
