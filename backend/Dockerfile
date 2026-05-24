# MathModel CoderAgent 代码执行沙箱
# 使用 python:3.12-slim 为基础镜像，安装数学建模常用库
FROM python:3.12-slim

# 安装数学建模核心库
RUN pip install --no-cache-dir \
    numpy pandas scipy matplotlib seaborn \
    scikit-learn xgboost statsmodels openpyxl \
    shap \
    && rm -rf /root/.cache

# 设置工作目录
WORKDIR /workspace

# 默认入口: shell，支持任意命令
# 用法: python /workspace/script.py 或 python -c "code"
CMD ["python3"]
