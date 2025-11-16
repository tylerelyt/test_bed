#!/bin/bash

# Python 环境快速配置脚本
# 功能：为项目配置固定的 Python 环境

echo "============================================================"
echo "Python 环境配置"
echo "============================================================"
echo ""

# 环境路径
PYTHON_ENV="/Users/tyler/miniconda3/envs/llama/bin/python"

echo "[INFO] 当前配置的 Python 环境:"
echo "       $PYTHON_ENV"
echo ""

# 验证环境
echo "[INFO] 验证 Python 版本..."
$PYTHON_ENV --version

echo ""
echo "[INFO] 检查已安装的关键包..."
$PYTHON_ENV -m pip list | grep -E "tensorflow|gradio|pandas|numpy|jieba" || echo "       部分包尚未安装"

echo ""
echo "============================================================"
echo "安装项目依赖"
echo "============================================================"
echo ""
echo "[PROMPT] 是否安装项目依赖？(y/n)"
read -r response

if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
    echo "[INFO] 开始安装依赖..."
    $PYTHON_ENV -m pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "[SUCCESS] 依赖安装完成"
    else
        echo ""
        echo "[ERROR] 依赖安装失败"
        echo "[HINT] 可能原因:"
        echo "  1. 磁盘空间不足 - 运行: conda clean --all"
        echo "  2. 网络问题 - 检查网络连接或配置代理"
        echo "  3. 权限问题 - 确保对环境目录有写权限"
    fi
else
    echo "[INFO] 跳过依赖安装"
fi

echo ""
echo "============================================================"
echo "配置完成"
echo "============================================================"
echo ""
echo "Cursor 已配置使用以下 Python 环境:"
echo "  $PYTHON_ENV"
echo ""
echo "下一步:"
echo "  1. 重启 Cursor 以加载新的环境配置"
echo "  2. 运行 ./quick_start.sh 启动项目"
echo "  3. 访问 http://localhost:7861"
echo ""

