# 使用 Python 官方映像檔作為基礎
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /base

# 複製要求的檔案到容器
COPY requirements.txt .

# 安裝所需的系統套件
RUN apt-get update && \
    apt-get install -y \
    zip \
    htop \
    screen \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安裝所需的 Python 套件
RUN pip install --no-cache-dir -r requirements.txt
