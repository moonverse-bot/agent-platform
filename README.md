# agent-platform

智能工单 / 客服 Agent 平台:FastAPI 后端 + 工具调用 Agent(OpenAI 兼容协议,默认接 DeepSeek)。

## 它解决什么问题

大模型只会「说」,不会「做」。这个项目给模型接上真实的业务能力:查工单、建工单、检索知识库,
并且把这些能力约束在**服务端**——鉴权、限流、数据隔离、执行轨迹都由后端负责,而不是交给前端或提示词。

一句话:把「调 API 的问答 demo」做成一个能上线的后端服务。

## 架构

```
HTTP 请求
   │
   ├─ trace 中间件        注入 trace_id,贯穿本次请求的所有日志
   │
   ├─ 依赖注入层          JWT 鉴权 → 当前用户 / 数据库会话 / 令牌桶限流 / Agent 运行时
   │
   ├─ 路由层              /api/auth  /api/tickets  /api/chat(SSE)
   │
   └─ Agent 运行时        提问 → 选择工具 → 执行 → 观察结果 → 作答(最多 N 轮)
          │
          └─ 工具             查工单 / 建工单 / 检索知识库
                 │
                 └─ 每次调用写 ToolCallLog(参数、结果预览、耗时、成功与否)
```

关键点:**工具是 Agent 触达业务的唯一通道**,所以参数的合法性、数据归属、异常兜底都在工具层收口。

## 目录结构

```
app/
├─ main.py                 应用装配:路由、中间件、生命周期
├─ config.py               配置(全部来自环境变量)
├─ core/
│  ├─ security.py          密码哈希 + JWT
│  ├─ ratelimit.py         令牌桶(Redis / 进程内双后端)
│  ├─ breaker.py           轻量熔断器(外部依赖连续失败时先断开一段时间)
│  ├─ logging.py           结构化 JSON 日志 + trace_id
│  └─ docs.py              中文接口文档页(向 Swagger UI 注入翻译脚本)
├─ db/
│  ├─ base.py              引擎与会话工厂
│  ├─ models.py            User / Ticket / Conversation / Message / ToolCallLog /
│  │                       KnowledgeArticle / KnowledgeChunk / KnowledgeVector
│  └─ init_db.py           建表 + 种子数据 + 整篇/段落两级向量回填
├─ schemas/                Pydantic 出入参
├─ api/
│  ├─ deps.py              依赖注入
│  ├─ auth.py              注册 / 登录 / 当前用户
│  ├─ tickets.py           工单 CRUD + 统计
│  ├─ chat.py              SSE 流式对话
│  └─ conversations.py     会话历史(列表 / 消息 / 删除)
├─ agent/
│  ├─ runner.py            Agent 执行循环(可流式)
│  ├─ llm.py               模型适配层
│  ├─ text.py              统一分词(英文整词 + 中文二元组),检索链路共用
│  ├─ embeddings.py        文本向量化(OpenAI 兼容服务 / 离线兜底)
│  ├─ chunking.py          文档切块(按段落累积,带标题前缀,可重叠)
│  ├─ retrieval.py         向量存储(pgvector / SQLite,整篇 + 段落)与 RRF 排名融合
│  ├─ rerank.py            重排(零依赖词法覆盖度 / OpenAI 兼容 rerank 服务)
│  ├─ memory.py            会话记忆
│  └─ tools/               工具定义与注册表
└─ web/                    零依赖前端页面(原生 HTML / CSS / JS)
tests/                     pytest,不联网、不依赖外部服务
```

## 快速开始

### 本地(SQLite,不需要装任何外部服务)

```powershell
cd D:\python-demo-ai-agent\agent-platform
"D:\pycharm\python 3.11.9\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env          # 然后填入 DEEPSEEK_API_KEY
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Windows 上也可以直接双击项目根目录的 `start.bat`:它会自动建虚拟环境、装依赖、缺 `.env` 时复制模板,
然后起服务并打开浏览器。

启动后:

- <http://127.0.0.1:8000> —— 聊天前端页面(推荐从这里开始)。
- <http://127.0.0.1:8000/docs> —— 中文接口文档,可以直接调接口。

> 知识库检索零配置就能跑(关键词 + 离线哈希向量)。想要真正的语义召回,把 `.env` 里的
> `EMBEDDING_API_KEY` 与 `EMBEDDING_MODEL` 填上即可,见下面「知识库检索」一节。

> `JWT_SECRET` 必须换成随机串且不少于 32 字节,否则 HS256 会打出 `InsecureKeyLengthWarning`。
>
> 文档页是全中文的:接口标题、分组说明、参数描述都用中文书写。Swagger UI 官方没有中文语言包,
> 所以 `app/core/docs.py` 在 FastAPI 生成的页面上注入了一段脚本,把「试一试 / 执行 / 响应」
> 这类界面文案替换掉;没覆盖到的文案会保持英文,不影响功能。

> 注意:一定要用 `.\.venv\Scripts\python.exe -m ...` 的形式调用工具。venv 里 `Scripts\*.exe`
> 这类启动器会把解释器路径写死在文件里,项目一旦移动目录就会失效。

### Docker(Postgres + Redis 完整栈)

```powershell
$env:DEEPSEEK_API_KEY="sk-xxxx"
docker compose up --build
```

> compose 里的 Postgres 用的是 `pgvector/pgvector:pg16`,不是官方镜像 —— 官方镜像不带 pgvector 扩展。
> 启动时会自动 `CREATE EXTENSION vector` 并建 HNSW 索引,不需要手工初始化。

## 部署上线

想把项目放到公网上给别人直接访问,见 **[DEPLOY.md](DEPLOY.md)**:

- 最省事:推到 GitHub → 在 Render 里一键 Blueprint(仓库已带 `render.yaml`)→ 填 `DEEPSEEK_API_KEY` → 拿到 `https://xxx.onrender.com`;
- 要数据不丢:再把 `DATABASE_URL` 换成 Neon 的免费 Postgres(自带 pgvector),启动时自动建表与回填向量;
- 自己的服务器:`docker compose up -d --build`。

> 公网部署前务必确认 `JWT_SECRET` 不是默认值、API Key 没有进仓库。免费实例的磁盘是临时的,
> 重启后访客数据会重置(知识库与演示账号会自动重建)。

## 前端页面

根路径 `/` 是一个零依赖的原生前端(HTML + CSS + JS,不引框架、不引 CDN),用浏览器打开就能走完一次完整的 Agent 对话:

- **登录 / 注册**:拿到 JWT 后存 `localStorage`,之后所有请求自动带 `Authorization` 头。
- **左侧会话列表**:历史会话可切换、可删除,点击即恢复聊天记录(数据来自 `/api/conversations`)。
- **中间聊天区**:逐字流式输出;模型调工具时会插一张「调用工具」卡片,显示工具名、参数、耗时和结果。
  若模型在调工具前先说了句废话,前端会按服务端的 `reset` 事件把那句撤掉,保证界面与最终答案一致。
- **右侧工单面板**:实时统计各状态工单数量,可直接新建工单、标记处理中、关闭。

页面对模型输出一律用 `textContent` 渲染,不用 `innerHTML`,避免模型生成的尖括号被当成 HTML 执行。

> 首次启动会创建一个演示账号 `demo@example.com` / `password-123`,并灌入 3 条示例工单和一段历史会话,
> 登录页也已经预填好这组信息 —— 打开页面点「登录」就能看到完整效果。不需要演示数据时,
> 把 `.env` 里的 `SEED_DEMO_DATA` 设为 `false`(线上部署**必须**关掉,否则任何人都能用已知密码登进来)。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 健康检查:返回模型是否启用、限流后端、检索模式 / 向量后端 / 当前 embedder、段落索引块数与重排器、已注册工具 |
| GET | `/` | 聊天前端页面(静态资源挂在 `/static`) |
| GET | `/docs` | 中文接口文档页 |
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/login` | 登录,返回 JWT |
| GET | `/api/auth/me` | 当前用户 |
| POST | `/api/tickets` | 建工单 |
| GET | `/api/tickets` | 工单列表(支持 `status` / `limit` / `offset`) |
| GET | `/api/tickets/stats` | 按状态统计 |
| GET | `/api/tickets/{id}` | 工单详情 |
| PATCH | `/api/tickets/{id}` | 改状态 / 优先级 |
| POST | `/api/chat` | Agent 对话,SSE 流式返回 |
| GET | `/api/conversations` | 当前用户的会话列表(按最近活跃倒序) |
| GET | `/api/conversations/{id}/messages` | 某会话内的消息记录 |
| DELETE | `/api/conversations/{id}` | 删除会话及其消息、工具调用轨迹 |

除注册登录、`/`、`/docs`、`/healthz` 外,其余接口都需要 `Authorization: Bearer <token>`。
带 `{id}` 的接口都会校验归属,拿别人的 id 只会得到 404。

## SSE 事件协议

`POST /api/chat` 返回 `text/event-stream`,事件顺序如下:

| 事件 | 载荷 | 含义 |
| --- | --- | --- |
| `meta` | `conversation_id` `trace_id` `remaining` | 会话已就绪 |
| `tool` | `name` `arguments` `ok` `latency_ms` `result` | 发生了一次工具调用 |
| `token` | `text` | 回答的增量文本 |
| `reset` | `reason` `discarded` | 丢弃本轮已下发的 token(调用工具前的铺垫文字) |
| `done` | `conversation_id` `content` `remaining` `limit` | 本次结束 |
| `error` | `message` | 出错(含未配置 API Key、模型调用失败等) |

前端可以直接用 `EventSource` 渲染「正在查资料 → 正在建单 → 逐字输出」的过程。

## 内置工具

| 工具 | 说明 |
| --- | --- |
| `create_ticket` | 创建工单,返回编号 |
| `list_my_tickets` | 列出当前用户的工单,可按状态过滤 |
| `get_ticket` | 按编号查详情 |
| `search_knowledge` | 检索知识库(关键词 + 整篇向量 + 段落向量三路召回,RRF 融合后重排) |

所有工具的查询都强制带 `owner_id`,模型无法越权读到其他用户的数据。

## 知识库检索

`search_knowledge` 不是单一策略,而是三路召回 → 融合 → 重排:

| 通道 | 做法 | 负责什么 |
| --- | --- | --- |
| 关键词 | 标题 ×3、标签 ×2、正文 ×1 加权,中文补一层二元组 | 字面命中:型号、编号、专有名词这类必须精确匹配的查询 |
| 整篇向量 | 整篇文档算一个向量,比余弦相似度 | 换个说法也能召回:用户问「钱怎么退回来」,文档里写的是「退款」 |
| 段落向量 | 文档切块后每块算一个向量,取「最好的那一块」代表整篇 | 长文档里只有一小句在讲这件事时,整篇向量会被其余内容稀释,段落向量不会 |

各路各取前 N 个候选,用 **RRF(Reciprocal Rank Fusion)** 融合:第 rank 名贡献 `1/(60+rank)` 分,
同一条结果把两路的分相加后重新排序。选 RRF 而不是加权求和,是因为关键词分是整数、
余弦是 0~1 的小数,两者量纲不可比,加权就得凭感觉调参;RRF 只看排名,不用调参,
也不会被某一路的分数量纲带跑。

整篇向量与段落向量的两路排名之间同样用 RRF 合并 —— 「整篇分」和「最好那一块的分」
也不是一个量纲,直接比大小没有意义。融合后的头部候选再交给重排器重排,
最后截到 `RETRIEVAL_TOP_K` 条返回。

### 文档切块

一篇文档只出一个向量时,标题、标签、正文会被平均成「一个点」:一篇 3000 字的说明里
也许只有一句在讲退款,它的向量会被其余 2900 字稀释掉,用户问退款时反而排不到前面。
所以入库时先把文档切成块、每块单独出一个向量,并且把**命中的那一块原文**作为
`matched_chunk` 一起返回 —— 模型拿到的是最贴题的那几句,而不是整篇。

切法刻意保持可解释,不引入模型:

1. 先按换行分段,段落是切块的最小单位,尽量不把一个操作步骤劈开;
2. 从前往后累积整段,超过 `KNOWLEDGE_CHUNK_SIZE` 就封口成一块;
3. 单段本身就超长(比如整段罗列条款)时按句末标点切开,单句还超长才硬切;
   硬切出来的碎片不再参与重叠,避免重叠逻辑把块越切越碎;
4. 相邻块之间保留 `KNOWLEDGE_CHUNK_OVERLAP` 个字符的重叠,别让答案正好掉进切口;
5. 每块前面带上文档标题 —— 块离开原文之后仍然有上下文。

参数取舍:块太小装不下一个完整答案,块太大一个向量里混的主题就多、精度就低。
默认 300 字 / 60 字重叠(中文里大致是一个完整操作步骤的体量)。把 `KNOWLEDGE_CHUNK_SIZE`
设为 0 就能关掉切块,退回「整篇一个向量」。

### 重排

召回为了快,用的是粗筛,顺序是近似的;重排只对头部少量候选(`RERANK_CANDIDATE_LIMIT`)
看得更细,性价比最高。默认实现是**词法覆盖度**:统计问题里的词有多少出现在候选文本里
(0~1 的分),覆盖度相同时再看这些词一共出现了多少次;段落级召回时,重排看的就是
命中的那一段,与召回保持同一粒度。

> 它零依赖、确定性强、分数可解释,但**它不是神经网络 cross-encoder**,同样理解不了同义词。
> 所以工具输出里一定带上重排器名字(`reranker` 字段),别把它的分数当成语义相关性。

想要真正的语义重排,把 `RERANK_PROVIDER` 改成 `api` 并填好 Key 与模型即可,代码不用动 ——
Cohere、硅基流动、Jina 的 `/rerank` 都是同一套协议:

```powershell
$env:RERANK_PROVIDER="api"
$env:RERANK_API_KEY="sk-xxxx"
$env:RERANK_BASE_URL="https://api.siliconflow.cn/v1"
$env:RERANK_MODEL="BAAI/bge-reranker-v2-m3"
```

重排是纯增强:关掉(`RERANK_ENABLED=false`)、候选只有一条、或者重排服务报错,
都会静默回到召回顺序,`rerank_score` 记为 `null`,检索本身不会失败。

**向量后端按数据库方言自动选**,同一套接口两个实现:

| 后端 | 用在 | 存储 | 检索方式 |
| --- | --- | --- | --- |
| `pgvector` | Postgres(`docker compose`) | 原生 `vector(N)` 列 | `<=>` 余弦距离 + HNSW 索引,检索在数据库里完成 |
| `sqlite-json` | SQLite(本地开发 / 测试) | JSON 文本列 | 进程内算余弦;数据量上来必须换 pgvector |

这里刻意没有引入 `pgvector` 的 Python 包:向量以文本参数传入,在 SQL 里 `CAST(:embedding AS vector)`,
这样只用 SQLite 的人不需要装任何新东西,`pip install -r requirements.txt` 也不会多出依赖。

段落级向量走同一套分工,只是表不同:SQLite 把段落向量和正文存在同一行(`knowledge_chunks`),
Postgres 则放在独立的 `knowledge_chunk_embeddings` 里,检索时 JOIN 回正文,省掉一次往返。
pgvector 的 HNSW 只支持单列,而不同 Embedding 模型的向量空间互不可比,
所以段落表的唯一索引与 HNSW 都带上 `WHERE embedder = ...`,按模型分区建。

### 打开真正的语义检索

不配任何东西也能跑:此时用的是 `local-hash-bigram-v2` —— 把切出来的词做特征哈希当向量,
只能匹配「词形重合」,**不是神经语义向量**。要具备真正的语义能力,填好下面两项即可,不用改代码:

```powershell
# 以硅基流动的 bge-m3 为例(OpenAI 兼容协议,1024 维)
$env:EMBEDDING_API_KEY="sk-xxxx"
$env:EMBEDDING_MODEL="BAAI/bge-m3"
$env:EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
$env:EMBEDDING_DIM="1024"
```

重启后启动日志会打印补算了多少条向量(幂等,已有向量的文章会跳过);
换了模型或维度,旧向量会自动视为过期并重算 —— `embedder` 名字与维度一起构成索引的版本。

> `EMBEDDING_DIM` 必须和模型输出维度一致,不一致会直接报错,而不是把脏向量写进索引。
> 用 pgvector 时改了维度要先 `DROP TABLE knowledge_embeddings` 与 `knowledge_chunk_embeddings`,
> 因为列类型是 `vector(N)`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

测试全程不联网:模型调用用假的客户端桩替换,数据库用临时 SQLite 文件。
Postgres 那条路径本机跑不了(没有 Docker),所以 `tests/test_retrieval.py` 退一步断言生成的 SQL:
距离算子、类型转换、HNSW 索引少一个都会在线上静默变慢或直接报错。

## 设计取舍

- **先显式循环,后图编排。** Agent 循环用普通 `for` 写,每一步都能打断点;等出现条件分支、并行工具、
  人工介入这些真实复杂度,再替换成 LangGraph,而不是为了用框架而用框架。
- **限流用令牌桶。** 比固定窗口更平滑,不会在窗口边界放行两倍流量;Redis 侧用 Lua 保证原子性,
  没配 Redis 时自动退化为进程内实现,本地开发不需要额外起服务。
- **工具异常转成文本。** 工具失败时把错误交给模型读,让它决定换工具还是如实告知用户,
  而不是直接 500 把整条对话打断。
- **流式 tool_calls 要拼接。** 模型返回的工具参数是按片段下发的,必须按 `index` 累积还原成完整 JSON,
  这是流式 + function calling 最容易踩的坑。
- **历史先读后写。** 落库本轮提问之前先取历史,否则当前这句会被当成历史重复带进上下文。
- **推理模型不下发 temperature。** `deepseek-reasoner` 不接受该参数,按模型决定是否携带。
- **调工具前的铺垫要 reset 掉。** 模型经常先吐一句「我这就去查」再发 tool_calls,
  这段已经按 token 推给前端了,但并不是最终答案。所以服务端补一个 `reset` 事件让前端丢弃,
  保证界面显示的内容与 `done.content` 落库的内容始终一致(提示词里也加了约束尽量减少这种情况)。
- **Windows 上必须用 `sqlalchemy[asyncio]`。** SQLAlchemy 给 `greenlet` 加的平台条件匹配的是
  `x86_64`,Windows 报的是 `AMD64`,条件不成立,裸装 sqlalchemy 会在用到异步扩展时才报缺 greenlet。
- **关键词与向量并存,不互相替代。** 纯向量会漏掉精确的字面命中(工单编号、型号),
  纯关键词接不住同义改写。所以做成多路召回 + RRF 融合,而不是二选一。
- **召回多粒度,单块得分不加总。** 同一篇文档既有整篇向量也有段落向量:整篇擅长判断
  「这篇是不是在讲这个主题」,段落擅长判断「这篇里有没有一句直接回答问题」。段落折算成
  文章时取「最好的那一块」而不是多块求和 —— 求和会让块多的长文档天然占优,
  排序就退化成「谁字多谁靠前」。
- **重排如实标注自己的能力边界。** 默认重排是词法覆盖度,不是神经网络 cross-encoder。
  所以工具输出里一定带上 `reranker` 名字与 `rerank_score`,让人一眼看出这个分数是怎么
  算出来的,而不是把词法分当成语义相关性去解读;要真正的语义重排就切到 rerank API。
- **切块的参数是取舍,不是最优解。** 块太小装不下一个完整答案,块太大一个向量里混的主题
  就多。默认 300 字 / 60 字重叠只是因为中文里它大致等于「一个完整操作步骤」,
  真实语料上应当按命中率调;所以它做成配置项,而不是写死在代码里。
- **熔断抽成独立组件,不写在业务里。** 向量检索是第一个会遇到「外部依赖挂了」的链路,
  但不会是最后一个(`app/core/breaker.py`)。把它从工具里拆出来,以后 rerank、LLM
  这些外部调用可以直接复用,也方便单独测试。
- **向量是增强链路,挂了不能拖垮主流程。** 没配模型、索引没建、模型服务 502,
  任何一种情况都只记一条状态(`embedding-failed` 等)然后退回关键词检索,
  `search_knowledge` 永远返回可用结果 —— 检索降级总好过整条对话挂掉。
  另外失败后会熔断 60 秒:否则每次提问都要先白等一次超时,一个挂掉的外部依赖
  会把所有对话一起拖慢。Embedding 的超时也压到 10 秒,因为这条链路是同步等结果的。
- **相似度阈值跟着 Embedding 模型走。** 离线哈希向量与神经向量的分数量纲完全不同
  (实测前者不相关 ≤0.06、命中 0.13~0.27;后者不相关也有 0.2~0.35),
  一个全局常量必然误杀其中一边,所以阈值定义在 embedder 自己身上,配置可覆盖。

## Roadmap

- **Phase 1(已完成)**:FastAPI 后端 + 工具调用 Agent + 会话历史接口 + 原生前端页面 + 覆盖以上链路的测试。
- **Phase 2(已完成)**:pgvector 语义检索与关键词结果融合排序,含 SQLite 便携后端、RRF 融合、
  相似度阈值与全套降级路径。
- **Phase 3(已完成)**:文档切块与段落级检索(整篇 + 段落双粒度、RRF 合并)、
  零依赖重排层与可切换的 rerank API、统一分词模块、熔断器抽成独立组件。
- **Phase 4**:Celery 异步任务(文档解析入库、批量工单);LangGraph 图编排;
  基于 ToolCallLog 的评测集与准确率看板;Token 成本统计。
