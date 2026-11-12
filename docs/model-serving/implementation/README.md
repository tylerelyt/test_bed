# 模型服务：代码与图解参考

[当前模块说明](../README.md) · [项目说明](../../../README.md)

原指南重复维护的启动步骤、能力列表和固定性能数字已经移出主说明。使用当前工作台时，先按模块 README 完成操作；需要理解旧设计时再展开下方示例。

## 当前实现入口

- [start_console.sh](../../../start_console.sh)
- [src/search_engine/workbench_api.py](../../../src/search_engine/workbench_api.py)
- [src/search_engine/model_service.py](../../../src/search_engine/model_service.py)
- [start_model_serving.py](../../../start_model_serving.py)

## 代码与图解参考

下列内容保留原有讲义中的代码、图解和输入输出示例。它们用于理解设计，可能省略初始化、依赖或使用旧接口；当前调用方式以本页“实现入口”链接的源码为准。示例中的指标、耗时和结果不作为本项目的验收数据。

<details>
<summary>展开代码与图解示例</summary>

### 示例 1

```mermaid
graph TB
    subgraph "客户端层"
        A[搜索界面] --> B[HTTP请求]
        C[训练界面] --> B
        D[外部系统] --> B
    end
    
    subgraph "API网关层"
        B --> E[Flask API Server<br/>端口:8501]
    end
    
    subgraph "业务逻辑层"
        E --> F[ModelService<br/>模型服务核心]
        F --> G[create_model_instance<br/>模型实例管理]
        F --> H[get_model_instance<br/>模型获取]
        F --> I[predict_ctr<br/>预测引擎]
    end
    
    subgraph "模型层"
        G --> J[CTRModel<br/>逻辑回归sklearn]
        G --> K[WideAndDeepCTRModel<br/>基于 TensorFlow 的 Wide & Deep 模型]
        H --> J
        H --> K
        I --> J
        I --> K
    end
```

### 示例 2

```python
# 文件路径: src/search_engine/model_service.py
class ModelService:
    """模型服务：负责模型训练、配置管理、模型文件等"""
    
    def __init__(self, model_file: str = None):
        if model_file is None:
            model_file = os.path.join(os.getcwd(), "models", "ctr_model.pkl")
        self.model_file = model_file
        self.ctr_model = CTRModel()  # 默认使用LR模型
        self.current_model_type = "logistic_regression"
        self.model_instances = {}  # 存储不同类型的模型实例
        self._load_model()
        
        # Flask API 服务相关
        self.flask_app = None
        self.api_running = False
```

### 示例 3

```python
def _setup_api_routes(self):
    """设置API路由"""
    
    @self.flask_app.route('/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({
            "status": "healthy",
            "model_type": self.current_model_type,
            "model_trained": self.ctr_model.is_trained
        })
    
    @self.flask_app.route('/v1/models', methods=['GET'])
    def list_models():
        """列出所有模型"""
        models = []
        for model_type in ['logistic_regression', 'wide_and_deep']:
            try:
                model_instance = self.get_model_instance(model_type)
                models.append({
                    "name": model_type,
                    "status": "loaded" if model_instance.is_trained else "unloaded",
                    "type": "pickle" if model_type == 'logistic_regression' else "tensorflow"
                })
            except:
                models.append({
                    "name": model_type,
                    "status": "error",
                    "type": "pickle" if model_type == 'logistic_regression' else "tensorflow"
                })
        
        return jsonify({"model": models})
    
    @self.flask_app.route('/v1/models/<model_name>/predict', methods=['POST'])
    def predict(model_name):
        """模型预测"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # 提取输入数据
            inputs = data.get('inputs', {})
            if not inputs:
                return jsonify({"error": "No inputs provided"}), 400
            
            # 执行预测
            ctr_score = self.predict_ctr(inputs, model_name)
            
            return jsonify({
                "outputs": {"ctr_score": ctr_score}
            })
            
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
```

### 示例 4

```bash
curl -X GET http://localhost:8501/health
```

### 示例 5

```json
{
    "status": "healthy",
    "model_type": "logistic_regression",
    "model_trained": true
}
```

### 示例 6

```bash
curl -X GET http://localhost:8501/v1/models
```

### 示例 7

```json
{
    "model": [
        {
            "name": "logistic_regression",
            "status": "loaded",
            "type": "pickle"
        },
        {
            "name": "wide_and_deep",
            "status": "loaded", 
            "type": "tensorflow"
        }
    ]
}
```

### 示例 8

```bash
curl -X GET http://localhost:8501/v1/models/logistic_regression
```

### 示例 9

```json
{
    "model": {
        "name": "logistic_regression",
        "status": "loaded",
        "type": "pickle"
    }
}
```

### 示例 10

```bash
curl -X POST http://localhost:8501/v1/models/logistic_regression:predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "query": "人工智能",
      "doc_id": "test_doc_001",
      "position": 1,
      "score": 0.8,
      "summary": "人工智能技术介绍"
    }
  }'
```

### 示例 11

```json
{
    "outputs": {
        "ctr_score": 0.123456
    }
}
```

### 示例 12

```bash
curl -X POST http://localhost:8501/v1/models/logistic_regression/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "query": "机器学习",
        "doc_id": "doc1",
        "position": 1,
        "score": 0.9,
        "summary": "机器学习介绍"
      },
      {
        "query": "深度学习", 
        "doc_id": "doc2",
        "position": 2,
        "score": 0.7,
        "summary": "深度学习介绍"
      }
    ]
  }'
```

### 示例 13

```json
{
    "outputs": [
        {
            "ctr_score": 0.234567
        },
        {
            "ctr_score": 0.345678
        }
    ]
}
```

### 示例 14

```python
def get_model_instance(self, model_type: str):
    """获取指定类型的模型实例"""
    # 每次都重新创建实例，确保加载最新的模型文件
    # 这解决了训练后模型不同步的问题
    self.model_instances[model_type] = self.create_model_instance(model_type)
    return self.model_instances[model_type]

def create_model_instance(self, model_type: str):
    """创建指定类型的模型实例"""
    try:
        if model_type in self.model_instances:
            return self.model_instances[model_type]
        
        if model_type == 'logistic_regression':
            from .training_tab.ctr_model import CTRModel
            model_instance = CTRModel()
            model_file = os.path.join(os.getcwd(), "models", "ctr_model.pkl")
        elif model_type == 'wide_and_deep':
            from .training_tab.ctr_wide_deep_model import WideAndDeepCTRModel
            model_instance = WideAndDeepCTRModel()
            model_file = os.path.join(os.getcwd(), "models", "wide_deep_ctr_model")
        else:
            raise ValueError(f"未实现的模型类型: {model_type}")
        
        model_instance.load_model(model_file)
        self.model_instances[model_type] = model_instance
        return model_instance
        
    except Exception as e:
        print(f"创建模型实例失败: {e}")
        # 回退到默认LR模型
        from .training_tab.ctr_model import CTRModel
        return CTRModel()
```

### 示例 15

```python
def _prepare_features(self, features: Dict[str, Any]) -> Optional[List[float]]:
    """准备特征向量"""
    try:
        # 这里需要根据实际的模型特征进行转换
        # 简化版本，实际应该根据训练时的特征工程逻辑
        feature_vector = []
        
        # 基本特征
        feature_vector.append(features.get('position', 1))
        feature_vector.append(features.get('score', 0.0))
        feature_vector.append(features.get('match_score', 0.0))
        feature_vector.append(features.get('query_ctr', 0.1))
        feature_vector.append(features.get('doc_ctr', 0.1))
        
        return feature_vector
    except Exception as e:
        print(f"特征准备失败: {e}")
        return None
```

### 示例 16

```python
def extract_features(self, ctr_data: List[Dict[str, Any]], is_training: bool = True, train_indices: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    从CTR数据中提取Wide和Deep特征（修复数据泄露问题）
    
    Args:
        ctr_data: CTR数据列表
        is_training: 是否为训练模式
        train_indices: 训练集索引（用于避免数据泄露）
    
    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray]: (特征字典, 标签数组)
    """
    # Wide特征：线性特征，如位置、文档长度等
    # Deep特征：高维特征，如文本嵌入、类别特征等
    # 具体实现请参考 ctr_wide_deep_model.py 中的 extract_features 方法
```

### 示例 17

```python
def check_and_start_model_service():
    """检查并启动模型服务（独立进程）"""
    # 1. 检查服务是否已运行
    model_service_url = "http://localhost:8501/health"
    try:
        req = request.Request(model_service_url, method="GET")
        with request.urlopen(req, timeout=2) as resp:
            if 200 <= resp.status < 300:
                print("✅ 检测到已运行的模型服务，直接复用")
                return True
    except Exception:
        pass
    
    # 2. 启动独立进程
    model_service_script = os.path.join(os.path.dirname(__file__), 'start_model_serving.py')
    process = subprocess.Popen(
        [sys.executable, model_service_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(__file__)
    )
    
    # 3. 等待服务启动并验证
    time.sleep(3)
    try:
        req = request.Request(model_service_url, method="GET")
        with request.urlopen(req, timeout=5) as resp:
            if 200 <= resp.status < 300:
                print("✅ 模型服务独立进程启动成功")
                return True
    except Exception as e:
        print(f"❌ 模型服务启动后健康检查失败: {e}")
        return False
```

### 示例 18

```bash
# 启动模型服务独立进程
python start_model_serving.py

# 后台启动
python start_model_serving.py &

# 使用进程管理器
python tools/model_service_manager.py start
python tools/model_service_manager.py stop
python tools/model_service_manager.py restart
python tools/model_service_manager.py status
python tools/model_service_manager.py health
```

### 示例 19

```python
#!/usr/bin/env python3
"""
启动Model Serving API服务
"""

import sys
import os
import signal

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from search_engine.model_service import ModelService

def signal_handler(signum, frame):
    """信号处理器"""
    print("\n🛑 收到停止信号，正在关闭服务...")
    sys.exit(0)

def main():
    """主函数"""
    print("🚀 启动Model Serving API服务...")
    print("=" * 50)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 创建并启动服务
        model_service = ModelService()
        
        print("📋 服务信息:")
        print(f"   地址: http://0.0.0.0:8501")
        print(f"   健康检查: http://localhost:8501/health")
        print(f"   模型列表: http://localhost:8501/v1/models")
        print("   按 Ctrl+C 停止服务")
        print("=" * 50)
        
        # 启动服务（这会阻塞进程）
        model_service.start_api_server(port=8501)
        
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")
    except Exception as e:
        print(f"❌ 启动服务失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 示例 20

```python
import sys
sys.path.append('src')
from search_engine.model_service import ModelService

# 创建并启动服务
model_service = ModelService()
model_service.start_api_server(port=8501)
```

### 示例 21

```python
# 默认配置
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8501
DEFAULT_DEBUG = False
DEFAULT_THREADED = True
```

### 示例 22

```python
# 模型文件路径
CTR_MODEL_PATH = "models/ctr_model.pkl"
WIDE_DEEP_MODEL_PATH = "models/wide_deep_ctr_model"  # 注意：没有.h5扩展名
WIDE_DEEP_H5_PATH = "models/wide_deep_ctr_model.h5"
WIDE_DEEP_TF_SERVING_PATH = "models/wide_deep_ctr_model_tf_serving/"

# 特征配置
FEATURE_DIMENSIONS = {
    'logistic_regression': 5,  # 实际是5维特征
    'wide_and_deep': 'variable'
}
```

### 示例 23

```python
    def start_api_server(self, host="0.0.0.0", port=8501, debug=False):
        """启动Flask API服务器（独立进程模式）"""
        try:
            if self.api_running:
                print("⚠️ API服务器已在运行")
                return True
            
            self.flask_app = Flask(__name__)
            self._setup_api_routes()
            
            self.api_running = True
            print(f"🚀 Model Serving API启动在 {host}:{port}")
            print("📋 可用接口:")
            print("   - 健康检查: http://localhost:8501/health")
            print("   - 模型列表: http://localhost:8501/v1/models")
            print("   - 预测接口: http://localhost:8501/v1/models/<model_name>/predict")
            print("   - 批量预测: http://localhost:8501/v1/models/<model_name>/batch_predict")
            print("=" * 50)
            
            # 直接运行Flask服务器（独立进程模式）
            self.flask_app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
            
        except Exception as e:
            print(f"❌ 启动API服务器失败: {e}")
            return False
```

### 示例 24

```python
@self.flask_app.route('/v1/models/<model_name>/batch_predict', methods=['POST'])
def batch_predict(model_name):
    """批量预测"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # 提取输入数据
        inputs_list = data.get('inputs', [])
        if not inputs_list:
            return jsonify({"error": "No inputs provided"}), 400
        
        # 执行批量预测
        results = []
        for inputs in inputs_list:
            ctr_score = self.predict_ctr(inputs, model_name)
            results.append({"ctr_score": ctr_score})
        
        return jsonify({
            "outputs": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### 示例 25

```python
class ModelService:
    def __init__(self, model_file: str = None):
        if model_file is None:
            model_file = os.path.join(os.getcwd(), "models", "ctr_model.pkl")
        self.model_file = model_file
        self.ctr_model = CTRModel()  # 默认使用LR模型
        self.current_model_type = "logistic_regression"
        self.model_instances = {}  # 存储不同类型的模型实例
        self._load_model()
        
        # Flask API 服务相关
        self.flask_app = None
        self.api_running = False
```

### 示例 26

```python
def get_memory_usage(self) -> Dict[str, float]:
    """获取内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
        "percent": process.memory_percent()
    }
```

### 示例 27

```python
def health_check(self) -> Dict[str, Any]:
    """服务健康检查"""
    return {
        "status": "healthy" if self.api_running else "unhealthy",
        "model_type": self.current_model_type,
        "model_trained": self.ctr_model.is_trained,
        "uptime": time.time() - self.start_time,
        "memory_usage": self.get_memory_usage()
    }
```

### 示例 28

```python
def model_health_check(self, model_type: str) -> bool:
    """模型健康检查"""
    try:
        model_instance = self.get_model_instance(model_type)
        if not model_instance.is_trained:
            return False
        
        # 测试预测
        test_inputs = {
            'query': 'test',
            'doc_id': 'test',
            'position': 1,
            'score': 0.5,
            'summary': 'test summary'
        }
        result = self.predict_ctr(test_inputs, model_type)
        return isinstance(result, (int, float)) and 0 <= result <= 1
    except Exception:
        return False
```

### 示例 29

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_serving.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@self.flask_app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")
```

### 示例 30

```python
import time

@self.flask_app.before_request
def before_request():
    request.start_time = time.time()

@self.flask_app.after_request
def after_request(response):
    duration = time.time() - request.start_time
    logger.info(f"Response time: {duration:.3f}s")
    return response
```

### 示例 31

```python
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "接口不存在"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "内部服务器错误"}), 500

@app.errorhandler(ValueError)
def value_error(error):
    return jsonify({"error": str(error)}), 400
```

### 示例 32

```python
def predict_ctr(self, features: Dict[str, Any], model_type: Optional[str] = None) -> float:
    """预测CTR"""
    try:
        # 始终使用指定类型的模型实例，确保使用最新训练的模型
        if model_type:
            model_instance = self.get_model_instance(model_type)
        else:
            # 如果没有指定模型类型，使用当前默认模型类型
            model_instance = self.get_model_instance(self.current_model_type)
        
        if not model_instance.is_trained:
            return 0.1  # 默认CTR
        
        # 使用指定模型的predict_ctr方法
        query = features.get('query', '')
        doc_id = features.get('doc_id', '')
        position = features.get('position', 1)
        score = features.get('score', 0.0)
        summary = features.get('summary', '')
        current_timestamp = features.get('timestamp')
        
        return model_instance.predict_ctr(query, doc_id, position, score, summary, current_timestamp)
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return 0.1  # 默认CTR值
```

### 示例 33

```python
import unittest
import requests

class TestModelServingAPI(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:8501"
    
    def test_health_check(self):
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
    
    def test_model_list(self):
        response = requests.get(f"{self.base_url}/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model", data)
    
    def test_predict(self):
        test_data = {
            "inputs": {
                "query": "人工智能",
                "doc_id": "test",
                "position": 1,
                "score": 0.8,
                "summary": "AI介绍"
            }
        }
        response = requests.post(
            f"{self.base_url}/v1/models/logistic_regression:predict",
            json=test_data
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("outputs", data)
```

### 示例 34

```python
import concurrent.futures
import time

def stress_test():
    """压力测试"""
    def make_request():
        test_data = {
            "inputs": {
                "query": "test",
                "doc_id": "test",
                "position": 1,
                "score": 0.5,
                "summary": "test"
            }
        }
        start_time = time.time()
        response = requests.post(
            "http://localhost:8501/v1/models/logistic_regression:predict",
            json=test_data
        )
        duration = time.time() - start_time
        return response.status_code, duration
    
    # 并发测试
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(100)]
        results = [future.result() for future in futures]
    
    # 统计结果
    success_count = sum(1 for status, _ in results if status == 200)
    avg_duration = sum(duration for _, duration in results) / len(results)
    
    print(f"成功率: {success_count/len(results)*100:.1f}%")
    print(f"平均响应时间: {avg_duration:.3f}s")
```

### 示例 35

```python
def test_end_to_end():
    """端到端测试"""
    # 1. 启动服务
    model_service = ModelService()
    model_service.start_api_server(port=8501)
    time.sleep(3)
    
    # 2. 健康检查
    response = requests.get("http://localhost:8501/health")
    assert response.status_code == 200
    
    # 3. 模型列表
    response = requests.get("http://localhost:8501/v1/models")
    assert response.status_code == 200
    
    # 4. 预测测试
    test_data = {
        "inputs": {
            "query": "人工智能",
            "doc_id": "test_doc",
            "position": 1,
            "score": 0.8,
            "summary": "AI技术介绍"
        }
    }
    response = requests.post(
        "http://localhost:8501/v1/models/logistic_regression:predict",
        json=test_data
    )
    assert response.status_code == 200
    
    # 5. 停止服务（独立进程会自动停止）
```

### 示例 36

```bash
# 查找占用端口的进程
lsof -i :8501

# 杀死进程
kill -9 <PID>

# 或使用不同端口
model_service.start_api_server(port=8502)
```

### 示例 37

```python
# 检查模型文件是否存在
import os
model_path = "models/ctr_model.pkl"
if not os.path.exists(model_path):
    print(f"模型文件不存在: {model_path}")

# 检查模型文件权限
if not os.access(model_path, os.R_OK):
    print(f"模型文件无读取权限: {model_path}")
```

### 示例 38

```python
def validate_prediction_result(result: float) -> bool:
    """验证预测结果"""
    if not isinstance(result, (int, float)):
        return False
    if not (0 <= result <= 1):
        return False
    return True
```

### 示例 39

```python
# 启用调试模式
model_service.start_api_server(port=8501, debug=True)
```

### 示例 40

```python
import logging

# 设置详细日志
logging.getLogger().setLevel(logging.DEBUG)
```

### 示例 41

```python
import cProfile
import pstats

def profile_prediction():
    """性能分析"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行预测
    result = model_service.predict_ctr(test_inputs, "logistic_regression")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### 示例 42

```python
class NewCTRModel:
    """新CTR模型实现"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def train(self, data):
        """训练模型"""
        pass
    
    def predict(self, features):
        """预测"""
        pass
```

### 示例 43

```python
class ModelService:
    def __init__(self):
        self.new_model = NewCTRModel()
    
    def get_model_instance(self, model_type: str):
        if model_type == 'new_model':
            return self.new_model
        # ... 其他模型
```

### 示例 44

```python
@self.flask_app.route('/v1/models/<model_name>/evaluate', methods=['POST'])
def evaluate_model(model_name):
    """模型评估接口"""
    try:
        data = request.get_json()
        # 实现评估逻辑
        return jsonify({"evaluation": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### 示例 45

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(features_hash: str, model_type: str) -> float:
    """缓存预测结果"""
    return self.predict_ctr(features, model_type)
```

### 示例 46

```python
import asyncio
import aiohttp

async def async_batch_predict(inputs_list, model_type):
    """异步批量预测"""
    tasks = []
    for inputs in inputs_list:
        task = asyncio.create_task(
            self.predict_ctr_async(inputs, model_type)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

</details>
