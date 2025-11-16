#!/bin/bash

# MLOps搜索引擎测试床 - 状态检查脚本
# 功能：检查系统各服务的运行状态

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
CYAN='\033[0;36m'
NC='\033[0m'

# 打印横幅
print_banner() {
    echo "============================================================"
    echo "MLOps搜索引擎测试床 - 系统状态检查"
    echo "============================================================"
}

# 检查单个端口状态
check_port_status() {
    local port=$1
    local service_name=$2
    
    if lsof -i :$port >/dev/null 2>&1; then
        local pid=$(lsof -ti :$port 2>/dev/null | head -1)
        local process_name=$(ps -p $pid -o comm= 2>/dev/null || echo "未知")
        echo -e "${GREEN}[RUNNING]${NC} $service_name (端口 $port) | PID: $pid | 进程: $process_name"
        return 0
    else
        echo -e "${RED}[STOPPED]${NC} $service_name (端口 $port)"
        return 1
    fi
}

# 检查所有服务状态
check_all_services() {
    echo ""
    echo "服务状态检查"
    echo "------------------------------------------------------------"
    
    local running=0
    local stopped=0
    
    # 定义端口和服务列表（bash 兼容方式）
    local ports=(7861 7860 3001 8501 7862 7863 7864 7865)
    local names=(
        "主界面 (Portal)"
        "Gradio 备用端口"
        "MCP 服务器"
        "模型服务 API"
        "数据服务"
        "索引服务"
        "扩展服务"
        "其他服务"
    )
    
    local i=0
    for port in "${ports[@]}"; do
        if check_port_status $port "${names[$i]}"; then
            running=$((running + 1))
        else
            stopped=$((stopped + 1))
        fi
        i=$((i + 1))
    done
    
    echo "------------------------------------------------------------"
    echo "统计: $running 个运行中, $stopped 个已停止"
    
    return $stopped
}

# 检查网络连接
check_network() {
    echo ""
    echo "网络连接检查"
    echo "------------------------------------------------------------"
    
    # 检查主界面
    if curl -s http://localhost:7861 >/dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} 主界面可访问: http://localhost:7861"
    else
        echo -e "${RED}[FAILED]${NC} 主界面不可访问"
    fi
    
    # 检查 MCP 服务
    if curl -s http://localhost:3001/health >/dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} MCP 服务可访问: http://localhost:3001"
    else
        echo -e "${YELLOW}[WARN]${NC} MCP 服务不可访问或无健康检查接口"
    fi
}

# 检查系统资源
check_resources() {
    echo ""
    echo "系统资源状态"
    echo "------------------------------------------------------------"
    
    # CPU 使用率（macOS）
    if command -v top >/dev/null 2>&1; then
        local cpu_usage=$(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' 2>/dev/null)
        if [ -n "$cpu_usage" ]; then
            echo "CPU 使用率: $cpu_usage"
        fi
    fi
    
    # 磁盘空间
    local disk_usage=$(df -h . | tail -1 | awk '{print $5}')
    local disk_avail=$(df -h . | tail -1 | awk '{print $4}')
    echo "磁盘使用: $disk_usage | 可用空间: $disk_avail"
}

# 显示快速操作提示
show_quick_actions() {
    echo ""
    echo "============================================================"
    echo "快速操作命令"
    echo "============================================================"
    echo "启动系统: ./quick_start.sh"
    echo "停止系统: ./stop_system.sh"
    echo "重启系统: ./restart_system.sh"
    echo "查看日志: tail -f logs/*.log"
}

# 主函数
main() {
    print_banner
    
    # 检查服务
    check_all_services
    local service_status=$?
    
    # 检查网络（可选）
    # check_network
    
    # 检查资源（可选）
    # check_resources
    
    # 显示快速操作
    show_quick_actions
    
    # 返回状态
    echo ""
    if [ $service_status -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] 系统运行正常${NC}"
        exit 0
    else
        echo -e "${YELLOW}[WARN] 部分服务未运行${NC}"
        exit 1
    fi
}

# 运行主函数
main "$@"

