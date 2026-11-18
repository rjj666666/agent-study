```
# ReAct Agent 带人工审批（Chroma RAG 演示版）
> 基于 LangGraph + LangChain + 智谱 GLM 实现的 ReAct Agent，**支持工具粒度的人工审批**，区分安全检索工具与高风险计算工具，RAG知识库检索免审批，仅计算器调用需要人工确认。
> ⚠️ 本版本为**学习验证 Demo，向量库使用 Chroma**，适合本地演示；商用交付版本已单独开发 Qdrant 版本（独立仓库），规避Chroma锁冲突、段错误等生产稳定性问题。

## ✨ 项目功能
- ✅ ReAct 智能体：LLM 自主思考、选择工具、多轮调用
- ✅ 工具分级审批：
  - 知识库检索工具 `rag_search`：**无需人工审批**，直接执行
  - 计算器工具 `calculator`：**强制人工审批**，输入 yes 才放行执行
- ✅ RAG 检索增强：支持 txt / pdf 文档加载、文本切分、向量检索、重排（CrossEncoder）
- ✅ 审计日志：所有审批操作自动写入 `logs/audit.log`，可追溯
- ✅ 结构化日志：程序运行日志统一输出到 `logs/agent.log`
- ✅ 状态持久化能力（LangGraph 原生）
- ✅ 接入智谱 GLM 系列大模型（兼容 OpenAI 协议）

## 📁 项目目录结构
```

agent_study/
├── config/                 # 配置模块
│   ├── **init**.py
│   └── settings.py         # 全局配置、环境变量读取
├── data/
│   └── knowledge.txt       # 默认知识库文件
├── docs/
│   └── README.md           # 项目文档
├── graph/                  # LangGraph 状态图核心
│   ├── **init**.py
│   ├── agent_graph.py      # 图构建、中断审批逻辑
│   └── state.py            # Agent 状态定义
├── llm/
│   ├── **init**.py
│   └── zhipu_llm.py        # 智谱 GLM LLM 实例封装
├── logs/                   # 日志目录（git 忽略）
│   ├── agent.log           # 运行日志
│   └── audit.log           # 审批审计日志
├── rag/                    # RAG 模块（向量库、加载、重排）
│   ├── **init**.py
│   ├── loader.py           # txt/pdf 文档加载 & 切片
│   ├── rerank.py           # CrossEncoder 结果重排
│   └── vector_store.py     # Chroma 向量库初始化、检索
├── tools/                  # 所有 Agent 工具统一管理
│   ├── **init**.py
│   ├── calculator.py       # 计算器工具（需要审批）
│   └── rag_search.py       # 知识库检索工具（免审批）
├── utils/                  # 工具函数
│   ├── **init**.py
│   ├── audit_log.py        # 审批日志写入函数
│   └── logger.py           # loguru 日志配置
├── chroma_db/              # Chroma 向量持久化目录（git 忽略）
├── .env                    # 环境密钥（git 忽略）
├── .gitignore
├── requirements.txt        # 依赖清单
├── run_agent.py            # 程序入口，启动对话
└── test.py                 # 测试脚本

```

## 🧰 技术栈
- 大模型：智谱 GLM（glm-5.3-flash），OpenAI兼容接口
- Agent框架：`langgraph==0.2.35`，状态图、human interrupt 断点审批
- 链框架：langchain
- 向量库：Chroma（本demo版）
- Embedding：BGE 本地嵌入模型
- Rerank：CrossEncoder 重排
- 日志：loguru
- 开发环境：WSL Ubuntu

## 🚀 快速启动
### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 .env 文件

在项目根目录新建 `.env`，填入你的智谱密钥：

```
OPENAI_API_KEY=你的智谱key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
MODEL_NAME=glm-5.3-flash

# 向量、重排模型配置
EMBED_MODEL_NAME=BAAI/bge-small-zh
RERANK_MODEL_NAME=BAAI/bge-reranker-base
RERANK_TOP_N=2
```

### 3. 运行项目

```
python run_agent.py
```

启动后输入问题，输入 `exit` 退出程序。

#### 测试示例

1. 提问：`Agent的开发路线` → 触发 RAG 检索，**直接返回结果，不需要审批**
2. 提问：`234*456等于` → 触发计算器，**弹出人工审批，输入 yes 执行**

## 📝 审计日志说明

所有计算器审批记录自动写入 `logs/audit.log`，记录：时间、工具名称、入参、审批结果，用于追溯审计。

## ⚠️ 重要说明 & 版本区分

1. **本仓库 = 学习演示版本，Chroma**
   - 优点：上手简单、代码直观，适合学习、面试展示迭代过程
   - 缺点：WSL / 服务器环境容易出现锁冲突、段错误，不适合直接交付给客户商用
2. **商用售卖版本（独立新项目）：Qdrant 版本**
   - 独立仓库，Docker 一键部署，自带 Web 向量管理面板，稳定性强、支持并发，面向私有化交付客户
3. 迭代思路：先使用 Chroma 快速验证 Agent + 分级审批的业务逻辑；验证稳定后，替换向量库为 Qdrant 作为生产交付成品