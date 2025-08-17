# 安装指南

## 系统要求

### 硬件要求
- **内存**: 至少2GB RAM
- **存储**: 至少1GB可用磁盘空间
- **CPU**: 双核以上处理器

### 软件要求
- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)

## 安装步骤

### 1. 环境准备

#### 检查Python版本
```bash
python --version
# 或
python3 --version
```

如果Python版本低于3.8，请先升级Python。

#### 创建虚拟环境（推荐）
```bash
# 使用venv
python -m venv search_env
source search_env/bin/activate  # Linux/macOS
# 或
search_env\Scripts\activate     # Windows

# 使用conda
conda create -n search_env python=3.10
conda activate search_env
```

### 2. 安装方式

#### 方式1: 从源码安装（推荐）
```bash
# 克隆项目
git clone https://github.com/tylerelyt/test_bed.git
cd intelligent-search-engine

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

#### 方式2: 使用pip安装
```bash
pip install intelligent-search-engine
```

#### 方式3: 最小化安装
```bash
# 只安装核心依赖
pip install gradio pandas numpy scikit-learn jieba matplotlib
```

### 3. 验证安装

```bash
# 检查安装
python -c "import search_engine; print('安装成功')"

# 启动系统
python start_system.py
```

访问 http://localhost:7861 查看界面。

## 可选组件

### 机器学习扩展
```bash
pip install intelligent-search-engine[ml]
```

### API服务
```bash
pip install intelligent-search-engine[api]
```

### 开发工具
```bash
pip install intelligent-search-engine[dev]
```

### 系统监控
```bash
pip install intelligent-search-engine[monitoring]
```

## 常见问题

### 1. 端口被占用
```bash
# 查看端口占用
lsof -i :7861  # macOS/Linux
netstat -ano | findstr :7861  # Windows

# 杀死占用进程
kill -9 <PID>  # macOS/Linux
taskkill /F /PID <PID>  # Windows
```

### 2. 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用清华源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 分别安装依赖
pip install gradio
pip install pandas
pip install scikit-learn
```

### 3. 中文分词问题
```bash
# 重新安装jieba
pip uninstall jieba
pip install jieba
```

### 4. 内存不足
- 减少batch_size参数
- 使用更小的模型
- 增加虚拟内存

### 5. 权限问题
```bash
# Linux/macOS
sudo pip install -r requirements.txt

# Windows
# 以管理员身份运行命令提示符
```

## 性能优化

### 1. 系统配置
```bash
# 设置环境变量
export PYTHONPATH=/path/to/project/src
export GRADIO_SERVER_PORT=7861
```

### 2. 内存优化
```python
# 在start_system.py中调整参数
data_service = DataService(
    auto_save_interval=60,  # 增加保存间隔
    batch_size=50          # 减少批量大小
)
```

### 3. 并发优化
```python
# 调整Gradio并发设置
interface.launch(
    server_port=7861,
    max_threads=10,        # 限制线程数
    show_error=True
)
```

## 开发环境设置

### 1. 代码格式化
```bash
pip install black flake8
black src/
flake8 src/
```

### 2. 类型检查
```bash
pip install mypy
mypy src/
```

### 3. 测试
```bash
pip install pytest
pytest test/
```

## 升级指南

### 升级到新版本
```bash
# 从源码升级
git pull origin main
pip install -r requirements.txt

# 从PyPI升级
pip install --upgrade intelligent-search-engine
```

### 数据迁移
```bash
# 备份数据
cp -r models/ models_backup/
cp -r data/ data_backup/

# 升级后恢复数据
# 检查数据格式兼容性
python tools/check_data_compatibility.py
```

## 卸载

```bash
# 卸载包
pip uninstall intelligent-search-engine

# 删除虚拟环境
conda env remove -n search_env
# 或
rm -rf search_env/

# 清理缓存
pip cache purge
```

## 技术支持

如果遇到安装问题，请：

1. 查看 [FAQ](FAQ.md)
2. 搜索 [Issues](https://github.com/tylerelyt/test_bed/issues)
3. 提交新的 [Issue](https://github.com/tylerelyt/test_bed/issues/new)
4. 联系维护者: tylerelyt@gmail.com 