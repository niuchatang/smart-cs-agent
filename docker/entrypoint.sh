#!/bin/sh
set -e
mkdir -p /app/data
# 命名卷首次挂载为空目录，需补回镜像内的知识库文件
if [ ! -f /app/data/knowledge_base.json ]; then
  cp /app/.defaults/knowledge_base.json /app/data/knowledge_base.json
fi
exec uvicorn main:app --host 0.0.0.0 --port 8010
