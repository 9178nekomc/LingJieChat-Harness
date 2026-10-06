# Nova Harness — 8M 迷你 MoE 模型的桌面推理客户端

一个为从零训练的 8M 参数 MiniMoE 模型打造的**推理增强层（Harness）+ 桌面聊天界面**，参考 DeepSeek 桌面客户端的交互风格用原创代码实现。

模型本身能力有限，Harness 用算法机制补齐短板：每条消息经过 **理解 → 拆解 → 路由 → 执行 → 门控** 的流水线，让它尽可能正常交流。

## 一键运行

```bash
pip install -r requirements.txt   # 只需要 torch
python app.py                     # 自动打开浏览器；装了 pywebview 则弹出原生桌面窗口
```

无需 GPU、无需联网、无需 API Key——模型权重就在 `checkpoints/` 里，克隆即用。

## Harness 补短板机制

| 小模型的短板 | Harness 机制 |
|---|---|
| 算术不可靠 | 独立计算引擎：自然语言算式解析（`what is 12 times 7`、`(2+3)*4`、`20% of 50`、平方/立方），AST 安全求值；也能接住模型自己发出的 `<tool>calc</tool>` 工具调用 |
| 连续多问就懵 | 问题自动拆解成子问题，逐个走对应通道后合并回答 |
| 忘性大 | 双层记忆：短期历史按 192 token 预算装填、旧轮次自动滚动摘要；长期记忆从对话抽取用户事实（名字/喜好等），localStorage 持久化，每次注入上下文 |
| 高频寒暄答不好 | 意图路由 + 高质量模板直达（问候/身份/致谢/道别/近况） |
| 复读机 / 退化输出 | 质量门控（三元组循环、空回复、脏话泄漏检测）→ 三级采样阶梯重试 → 问题改写 → 模板兜底 |
| 不友好输入 | 安全策略：拒答并重置对话 |

界面亮点：流式气泡输出、可折叠的 **Harness 思考过程**面板（完整展示拆解/记忆/重试/工具调用链路）、意图与重试标签、底部用量看板（tok/s / 重试次数 / 上下文占用 `x/192`）、长期记忆卡片、三套皮肤（深空 / 极光 / 晨雾）。

## 目录结构

```
├── app.py              # 桌面入口（浏览器 / pywebview 原生窗口）
├── config.py           # 8M MoE + LoRA 模型配置
├── model.py            # MiniMoE 模型定义（MoE + GQA + RMSNorm + SwiGLU + LoRA）
├── vocab.py            # 字符/词混合 tokenizer
├── requirements.txt
├── harness/            # 推理增强层
│   ├── engine.py       #   模型加载 + 流式生成 + 上下文预算
│   ├── router.py       #   意图识别 + 问题拆解 + 改写
│   ├── math_skill.py   #   数学计算引擎
│   ├── user_memory.py  #   长期记忆抽取
│   ├── templates.py    #   模板回复库
│   ├── pipeline.py     #   流水线编排（SSE 事件流）
│   └── server.py       #   本地 HTTP 服务（静态页 + SSE API）
├── web/                # 前端（原生 JS，无框架）
└── checkpoints/        # 训练好的权重（pretrain.pt + lora.pt + tokenizer.json）
```

## 模型架构（约 8M 参数）

4 层 Transformer decoder · dim=256 · 8 头 GQA（4 KV 头）· SwiGLU · RMSNorm · 4 专家 top-2 MoE · 上下文 192 token · LoRA (r=16) 微调 · CPU 推理。

模型由同仓库的训练流程从零训练而来（预训练 + LoRA SFT，英文对话 + 计算器工具调用数据），训练代码见主仓库 `minimoe/`。

## 架构说明

- **服务端零写盘**：会话与长期记忆由浏览器 localStorage 持久化，随请求回传；本地 HTTP 服务无状态，可随时重启。
- **SSE 流式**：模型逐 token 生成，前端实时渲染；重试时前端自动清空气泡重来。
- **强制 CPU**：小模型单核即可流畅推理（约 20–40 tok/s）。

## License

MIT
