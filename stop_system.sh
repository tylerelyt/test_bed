#!/bin/bash

# MLOps搜索引擎测试床 - 停止脚本
# 功能：优雅地停止系统的所有服务

# 优先使用 zsh，但默认兼容 bash
if [ -z "$RUNNING_IN_ZSH" ] && [ -n "$ZSH_VERSION" ]; then
    :  # Already running in zsh
elif [ -z "$RUNNING_IN_ZSH" ] && command -v zsh >/dev/null 2>&1 && [ -t 0 ]; then
    export RUNNING_IN_ZSH=1
    exec zsh "$0" "$@"
fi

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 打印横幅
print_banner() {
    echo "============================================================"
    echo "MLOps搜索引擎测试床 - 系统停止脚本"
    echo "============================================================"
}

# 停止指定端口的进程
stop_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null)
    
    if [ -n "$pids" ]; then
        echo "[INFO] 正在停止端口 $port 上的进程..."
        
        # 先尝试优雅停止 SIGTERM
        for pid in $pids; do
            if kill -0 $pid 2>/dev/null; then
                echo "[DEBUG] 发送 SIGTERM 到进程 $pid"
                kill -TERM $pid 2>/dev/null || true
            fi
        done
        
        # 等待 5 秒让进程自行退出
        sleep 5
        
        # 检查是否还有残留进程
        pids=$(lsof -ti :$port 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "[WARN] 进程未响应，强制停止..."
            for pid in $pids; do
                if kill -0 $pid 2>/dev/null; then
                    kill -9 $pid 2>/dev/null || true
                fi
            done
            sleep 1
        fi
        
        # 最终检查
        if lsof -i :$port >/dev/null 2>&1; then
            echo -e "${RED}[ERROR] 端口 $port 停止失败${NC}"
            return 1
        else
            echo -e "${GREEN}[SUCCESS] 端口 $port 已停止${NC}"
            return 0
        fi
    else
        echo "[INFO] 端口 $port 未被占用"
        return 0
    fi
}

# 停止所有服务
stop_all_services() {
    echo ""
    echo "正在停止所有服务..."
    echo "------------------------------------------------------------"
    
    # 定义系统使用的端口
    local ports=(7860 7861 7862 7863 7864 7865 3001 8501)
    local failed=0
    
    for port in "${ports[@]}"; do
        if ! stop_port $port; then
            failed=$((failed + 1))
        fi
    done
    
    echo "------------------------------------------------------------"
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] 所有服务已停止${NC}"
        return 0
    else
        echo -e "${RED}[ERROR] 有 $failed 个服务停止失败${NC}"
        return 1
    fi
}

# 主函数
main() {
    print_banner
    
    if stop_all_services; then
        echo ""
        echo "系统已成功停止"
        exit 0
    else
        echo ""
        echo "[ERROR] 系统停止过程中出现错误"
        echo "[HINT] 建议：检查错误信息，或使用 './status_system.sh' 查看状态"
        exit 1
    fi
}

# 运行主函数
main "$@"
