FROM python:3.13-slim

WORKDIR /app

# 安装基础系统依赖（用于编译部分Python包，如果不需要可以删掉）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（必须包含官方评分需要的库）
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir numpy pandas pyarrow tomli requests statsmodels scipy

# 复制项目代码（注意：这一步会复制当前目录下的所有文件）
# 虽然会复制 mock_api.py，但我们通过下面的 ENTRYPOINT 确保绝对不会运行它
COPY . .

# 设置容器入口（官方会执行：docker run ... forecast --panels ...）
# 这里的入口设为 python my_agent.py，我们的代码会自动处理后面的 forecast 参数
ENTRYPOINT ["python", "my_agent.py"]
CMD ["forecast"]