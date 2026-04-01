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

EXPOSE 8010
ENTRYPOINT ["/entrypoint.sh"]
