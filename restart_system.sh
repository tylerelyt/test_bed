#!/bin/bash

# MLOps搜索引擎测试床 - 重启脚本
# 功能：先停止系统，再重新启动

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
    echo "MLOps搜索引擎测试床 - 系统重启脚本"
    echo "============================================================"
}

# 主函数
main() {
    print_banner
    
    # 步骤1：停止系统
    echo ""
    echo "步骤 1/2: 停止当前系统"
    echo "------------------------------------------------------------"
    
    if [ -f "./stop_system.sh" ]; then
        if ./stop_system.sh; then
            echo -e "${GREEN}[SUCCESS] 系统已停止${NC}"
        else
            echo -e "${RED}[ERROR] 停止系统失败${NC}"
            echo "[PROMPT] 是否继续启动？(输入 'yes' 继续)："
            read -r response
            if [ "$response" != "yes" ]; then
                echo "[INFO] 重启已取消"
                exit 1
            fi
        fi
    else
        echo "[WARN] 未找到 stop_system.sh，跳过停止步骤"
    fi
    
    # 等待端口释放
    echo ""
    echo "[INFO] 等待端口释放..."
    sleep 3
    
    # 步骤2：启动系统
    echo ""
    echo "步骤 2/2: 启动系统"
    echo "------------------------------------------------------------"
    
    if [ -f "./quick_start.sh" ]; then
        exec ./quick_start.sh
    else
        echo "[ERROR] 未找到 quick_start.sh"
        echo "[INFO] 尝试直接启动..."
        
        if [ -f "./start_system.py" ]; then
            python start_system.py
        else
            echo -e "${RED}[ERROR] 未找到启动脚本${NC}"
            exit 1
        fi
    fi
}

# 运行主函数
main "$@"
