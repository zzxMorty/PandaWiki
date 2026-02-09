# PandaWiki 访问指南

## 前端访问地址

### 1. 管理端（Admin Console）
**地址**: http://localhost:5173/

- 登录页面：http://localhost:5173/login
- 用于管理知识库、用户、模型配置等

### 2. 用户端（Wiki 网站）
**地址**: http://localhost:3010/

- 面向最终用户的 Wiki 网站前台
- 需要先创建知识库后才能访问

---

## 登录账号设置

### 方式一：通过配置创建默认管理员（推荐）

在 `backend/config.yml` 中设置 `admin_password`，或设置环境变量：

```bash
export ADMIN_PASSWORD=your_password_here
```

然后重启后端 API，系统会自动创建/更新账号：
- **账号**: `admin`
- **密码**: 你设置的 `ADMIN_PASSWORD` 值

### 方式二：通过 API 创建第一个用户

如果 `admin_password` 为空，可以通过 API 创建第一个管理员：

```bash
curl -X POST http://localhost:8000/api/v1/user/create \
  -H "Content-Type: application/json" \
  -d '{
    "account": "admin",
    "password": "your_password_here",
    "role": "admin"
  }'
```

**注意**: 密码需至少 8 位。

---

## 快速开始

1. **设置默认管理员密码**（在 `backend/config.yml` 或环境变量）：
   ```yaml
   admin_password: "your_password_here"
   ```

2. **重启后端 API**（如果已在运行）：
   ```bash
   pkill -f panda-wiki-api
   cd backend
   /tmp/panda-wiki-api  # 或 go run ./cmd/api
   ```

3. **访问管理端**：
   - 打开 http://localhost:5173/login
   - 账号：`admin`
   - 密码：你设置的密码

4. **首次登录后**：
   - 配置 AI 模型（Chat 模型）
   - 创建知识库
   - 开始使用 PandaWiki

---

## 服务状态检查

- **后端 API**: http://localhost:8000/api/v1/user/login（POST 请求返回 200 即正常）
- **管理端**: http://localhost:5173/
- **用户端**: http://localhost:3010/
