# ---------- Stage 1: 构建 Vue 前端 → static-vue ----------
FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: FastAPI 应用 ----------
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY static ./static
COPY data/knowledge_base.json ./.defaults/knowledge_base.json
COPY --from=frontend-build /app/static-vue ./static-vue
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && mkdir -p /app/data

# 内置默认环境变量：docker pull 后无需挂载 .env 即可配合 compose 中的 MySQL 使用
# （compose 仍会覆盖 MYSQL_HOST 等为容器网络地址）
ENV AMAP_API_KEY=ef32862c683c56ee5955813466992474 \
    DEEPSEEK_API_KEY=sk-d77fdfeb61a0476dbeabfd26d72bab00 \
    DEEPSEEK_BASE_URL=https://api.deepseek.com \
    DEEPSEEK_MODEL_ID=deepseek-chat \
    MYSQL_HOST=mysql \
    MYSQL_PORT=3306 \
    MYSQL_USER=root \
    MYSQL_PASSWORD=156718 \
    MYSQL_DATABASE=smart_cs_agent \
    MYSQL_UNIX_SOCKET=

EXPOSE 8010
ENTRYPOINT ["/entrypoint.sh"]
