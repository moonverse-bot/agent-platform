# 部署上线指南

目标:让这个项目有一个公网地址,别人打开浏览器就能用(聊天、查工单、检索知识库)。

## 先选方案

| 方案 | 花钱 | 你需要准备 | 说明 |
| --- | --- | --- | --- |
| **A. Render(推荐)** | 免费 | GitHub 账号、Render 账号、DeepSeek Key | 5~10 分钟出地址;闲置会休眠、磁盘是临时的 |
| B. Render + Neon | 免费 | 同上 + Neon 账号 | 数据不会因为重启丢失 |
| C. 自己的服务器 | 服务器费用 | 一台云服务器 + Docker | 最灵活;国内节点绑域名要备案 |

仓库里已经放好 `render.yaml`,选方案 A 基本是「点几下」的事。

---

## 方案 A:Render 免费部署

### 0. 准备三样东西

1. GitHub 账号(存代码)
2. Render 账号(https://dashboard.render.com ,用 GitHub 登录即可)
3. DeepSeek API Key(你本机系统环境变量里已经有的那个 `DEEPSEEK_API_KEY`)

### 1. 把代码推到 GitHub

项目现在还不是 Git 仓库,在项目目录里执行:

```powershell
cd D:\python-demo-ai-agent\agent-platform
git init
git add .
git commit -m "feat: agent platform with hybrid retrieval"
```

然后在 GitHub 网页上新创建一个**空仓库**(不要勾选 README / .gitignore),再执行:

```powershell
git remote add origin https://github.com/你的用户名/agent-platform.git
git branch -M main
git push -u origin main
```

> `.env`、`*.db`、`.venv` 都已经写在 `.gitignore` 里,不会被推上去。
> **永远不要把 API Key 写进代码或提交到仓库**,Key 只放在平台的环境变量里。

### 2. 在 Render 创建服务

1. 打开 https://dashboard.render.com
2. 右上角 **New +** → **Blueprint**
3. 选刚才的仓库 → **Apply**
4. 它会读到 `render.yaml`,并让你填 `DEEPSEEK_API_KEY`(直接粘贴你的 Key)
5. 点 **Create / Deploy**,等 2~5 分钟

### 3. 打开公网地址

部署完成后地址形如:

```
https://agent-platform-xxxx.onrender.com
```

- 首页就是聊天界面
- 演示账号:`demo@example.com` / `password-123`
- 健康检查:`https://agent-platform-xxxx.onrender.com/healthz`
- 接口文档:`https://agent-platform-xxxx.onrender.com/docs`

可以先用演示账号问一句「我有哪些工单?」,能看到工具调用过程就说明整条链路通了。

### 4. 免费版必须知道的三个限制

- **会休眠**:15 分钟没人访问就暂停,下次打开要等 30~60 秒才醒。发给面试官时可以先自己点一下把它叫醒。
- **磁盘是临时的**:每次重新部署/重启,数据库会回到初始状态。知识库和演示账号会自动重建,但访客创建的工单和对话会丢。
- **每月 750 小时**:单个免费服务够用。

---

## 方案 B:换成 Postgres(数据不丢,仍然免费)

Render 的免费 Web 服务配 [Neon](https://neon.tech) 的免费 Postgres(自带 pgvector)。

1. Neon 注册 → **New Project**,区域选 Singapore
2. 复制它给的连接串,形如:

```
postgresql://user:password@ep-xxx.ap-southeast-1.aws.neon.tech/neondb?sslmode=require
```

3. 把开头的 `postgresql://` 改成 `postgresql+asyncpg://`(项目用异步驱动),得到:

```
postgresql+asyncpg://user:password@ep-xxx.ap-southeast-1.aws.neon.tech/neondb?sslmode=require
```

4. Render → 你的服务 → **Environment** → 添加:

```
DATABASE_URL = postgresql+asyncpg://...（上一步那串）
```

5. 点 **Manual Deploy** 重新部署
6. 打开 `/healthz`,确认 `vector_backend` 变成 `pgvector`

启动时会自动执行 `CREATE EXTENSION IF NOT EXISTS vector`、建表和回填向量,不需要手工执行 SQL。

> 如果日志里报 `connect() got an unexpected keyword argument 'sslmode'`,
> 把连接串末尾的 `sslmode=require` 换成 `ssl=require` 即可。

---

## 方案 C:自己的服务器

代码里已经有 `Dockerfile` 和 `docker-compose.yml`(Postgres 用的是带 pgvector 的镜像):

```bash
git clone <你的仓库地址>
cd agent-platform
export DEEPSEEK_API_KEY=sk-xxxx
export JWT_SECRET=$(openssl rand -hex 32)
docker compose up -d --build
```

然后访问 `http://服务器IP:8000`。

- 国内云服务器(阿里云/腾讯云)**大陆节点**绑定域名需要备案;只是演示可以先用 `IP:8000`。
- 最低配置 1 核 2G 就够;SQLite 数据在容器重建后会丢,生产请用 compose 里的 Postgres。
- 前面如果加了 Nginx,记得 `proxy_buffering off;`,否则打字机效果会变成「一次性吐完」。

---

## 上线前检查清单

- [ ] `JWT_SECRET` 是随机串(`render.yaml` 已设置为自动生成)
- [ ] `DEEPSEEK_API_KEY` 只存在于平台环境变量,仓库里没有
- [ ] `/healthz` 返回 `"llm_enabled": true`
- [ ] 用演示账号完整问一句,能看到工具调用 + 流式回答
- [ ] 如果不想公开演示账号,改 `app/db/init_db.py` 里的 `DEMO_PASSWORD`,
      并同步改 `app/web/index.html` 里预填的密码

## 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 第一次打开特别慢 | 免费实例在休眠,等 30~60 秒;之后会正常 |
| 回答「调用大模型失败: Connection error」 | `DEEPSEEK_API_KEY` 没填或填错;去 Render 的 Environment 改完重新部署 |
| 页面能打开但接口 404 | 访问的是被改写过的子路径;请直接用根域名访问 |
| 回答一次性出现,没有打字效果 | 中间代理开了缓冲;Render 不用额外配置,自建 Nginx 要 `proxy_buffering off` |
| 部署失败 | 看 Render 的 **Logs**;常见是 Python 版本不对(仓库里有 `.python-version`)或启动命令没监听 `$PORT` |
| 重启后工单没了 | 免费实例磁盘是临时的;按方案 B 换成 Postgres |
