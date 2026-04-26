import gradio as gr
import sys
import os

# 修复 Gradio API 信息生成时的类型推断错误
# 这个问题发生在某些组件的 JSON schema 中 additionalProperties 是 bool 而不是 dict
def _patch_gradio_api_info():
    """修复 Gradio API 信息生成时的类型推断错误"""
    try:
        from gradio_client import utils as client_utils
        
        # 保存原始函数
        original_get_type = client_utils.get_type
        
        def patched_get_type(schema):
            """修复后的 get_type 函数，处理 bool 类型的 schema"""
            # 如果 schema 是 bool，返回默认类型
            if isinstance(schema, bool):
                return "Any"
            # 如果 schema 是 dict 但缺少必要的键，返回默认类型
            if not isinstance(schema, dict):
                return "Any"
            # 检查 "const" 键是否存在（原始代码会在这里出错）
            if "const" in schema:
                return original_get_type(schema)
            # 其他情况调用原始函数
            return original_get_type(schema)
        
        # 替换函数
        client_utils.get_type = patched_get_type
        
        # 同样修复 _json_schema_to_python_type 函数
        original_json_schema_to_python_type = client_utils._json_schema_to_python_type
        
        def patched_json_schema_to_python_type(schema, defs=None):
            """修复后的 _json_schema_to_python_type 函数"""
            # 如果 schema 是 bool，返回 "Any"
            if isinstance(schema, bool):
                return "Any"
            # 如果 additionalProperties 是 bool，将其转换为 dict
            if isinstance(schema, dict) and "additionalProperties" in schema:
                if isinstance(schema["additionalProperties"], bool):
                    # 如果是 True，表示允许任意属性，返回 "Dict[str, Any]"
                    # 如果是 False，表示不允许额外属性，返回原始类型
                    if schema["additionalProperties"]:
                        return "Dict[str, Any]"
                    else:
                        # 移除 additionalProperties，继续处理
                        schema = schema.copy()
                        schema.pop("additionalProperties")
            return original_json_schema_to_python_type(schema, defs)
        
        # 替换函数
        client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
        
    except Exception as e:
        print(f"⚠️  修复 Gradio API 信息生成失败: {e}")

# 在导入其他模块之前应用修复
_patch_gradio_api_info()

from .index_tab import build_index_tab
from .search_tab import build_search_tab
from .training_tab import build_training_tab
from .monitoring_tab import build_monitoring_tab
from .rag_tab import build_rag_tab
from .mcp_tab import build_mcp_tab
from .image_tab.image_tab import build_image_tab
from .service_manager import service_manager

class SearchUI:
    def __init__(self):
        # 使用服务管理器
        print("🚀 启动服务管理器...")
        self.service_manager = service_manager
        
        # 获取服务实例
        self.data_service = self.service_manager.data_service
        self.index_service = self.service_manager.index_service
        self.model_service = self.service_manager.model_service
        self.image_service = self.service_manager.image_service
        
        self.current_query = ""
        self.setup_ui()

    def setup_ui(self):
        with gr.Blocks(title="搜索引擎测试床 - 服务架构版本") as self.interface:
            gr.Markdown("""
            # 🔬 搜索引擎测试床 - 服务架构版本
            
            ## 🏗️ 系统架构
            - **数据服务 (DataService)**: CTR事件收集、样本状态管理
            - **索引服务 (IndexService)**: 索引构建、文档管理、检索功能
            - **模型服务 (ModelService)**: 模型训练、配置管理、模型文件
            - **RAG服务 (RAGService)**: 直连LLM / 检索增强 / OAG图谱游走推理
            - **上下文工程服务 (MCPService)**: 符号主义专家系统 + 连接主义LLM消费 (v2.0)
            - **多模态服务 (ImageService)**: 图片检索(CLIP)、图像生成(Stable Diffusion)
            
            ## 📊 服务状态
            - 数据服务: ✅ 运行中
            - 索引服务: ✅ 运行中
            - 模型服务: ✅ 运行中
            - RAG服务: ✅ 运行中
            - 上下文工程服务: ✅ 运行中 (v2.0完整架构)
            - 图片服务: ✅ 运行中 (基于CLIP模型)
            """)
            
            with gr.Tabs():
                with gr.Tab("🏗️ 第一部分：离线索引构建"):
                    build_index_tab(self.index_service)
                with gr.Tab("🔍 第二部分：在线召回排序"):
                    build_search_tab(self.index_service, self.data_service)
                with gr.Tab("🤖 第三部分：RAG"):
                    build_rag_tab(self.index_service, page_mode="rag")
                with gr.Tab("🧠 第四部分：OAG（图谱游走推理）"):
                    build_rag_tab(self.index_service, page_mode="oag")
                with gr.Tab("🧩 第五部分：上下文工程"):
                    build_mcp_tab()
                with gr.Tab("🖼️ 第六部分：多模态系统"):
                    build_image_tab(self.image_service)
                with gr.Tab("📊 第七部分：数据回收训练"):
                    build_training_tab(self.model_service, self.data_service)
                with gr.Tab("🛡️ 第八部分：系统监控"):
                    build_monitoring_tab(self.data_service, self.index_service, self.model_service)

    def run(self):
        port = 7861  # 修改端口避免冲突
        print(f"✅ 搜索引擎测试床 UI 启动：http://localhost:{port}")
        print(f"📊 数据服务状态: 运行中 (共{len(self.data_service.get_all_samples())}条CTR样本)")
        print(f"📄 索引服务状态: 运行中 (共{self.index_service.get_stats()['total_documents']}个文档)")
        
        model_info = self.model_service.get_model_info()
        if model_info['is_trained']:
            print(f"🤖 模型服务状态: 运行中 (已训练模型)")
        else:
            print(f"🤖 模型服务状态: 运行中 (未训练)")
        
        image_stats = self.image_service.get_stats()
        print(f"🖼️ 图片服务状态: 运行中 (共{image_stats['total_images']}张图片，{image_stats['model_device']}设备)")
        
        try:
            # 已经通过 _patch_gradio_api_info() 修复了 API 信息生成的错误
            self.interface.launch(share=False, inbrowser=True, server_port=port)
        except Exception as e:
            # 如果 show_api=False 不支持，尝试捕获 API 信息生成错误
            error_str = str(e)
            if "additionalProperties" in error_str or "bool" in error_str or "TypeError" in error_str:
                print(f"⚠️  API 信息生成时出现类型推断错误（不影响使用）: {type(e).__name__}")
                # 尝试不显示 API 信息
                try:
                    self.interface.launch(share=False, inbrowser=True, server_port=port, show_api=False)
                except:
                    # 如果还是失败，尝试捕获并继续
                    import threading
                    def run_with_error_handling():
                        try:
                            self.interface.launch(share=False, inbrowser=False, server_port=port, show_api=False)
                        except Exception as e3:
                            if "additionalProperties" not in str(e3) and "bool" not in str(e3):
                                raise
                    thread = threading.Thread(target=run_with_error_handling, daemon=True)
                    thread.start()
                    import time
                    time.sleep(2)
            else:
                print(f"❌ 启动失败: {e}")
                # 尝试其他端口
                for alt_port in [7862, 7863, 7864, 7865]:
                    try:
                        print(f"🔄 尝试端口 {alt_port}...")
                        self.interface.launch(share=False, inbrowser=True, server_port=alt_port, show_api=False)
                        break
                    except Exception as e2:
                        if "additionalProperties" not in str(e2) and "bool" not in str(e2):
                            print(f"❌ 端口 {alt_port} 也失败: {e2}")
                            continue

def main():
    ui = SearchUI()
    ui.run()

if __name__ == "__main__":
    main()
