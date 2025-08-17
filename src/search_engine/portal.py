import gradio as gr
import sys
import os

from .index_tab import build_index_tab
from .search_tab import build_search_tab
from .training_tab import build_training_tab
from .monitoring_tab import build_monitoring_tab
from .rag_tab import build_rag_tab
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
            - **上下文工程服务 (RAGService)**: 直连LLM / 检索增强 / 多步推理 (Ollama)
            - **图片服务 (ImageService)**: 基于CLIP的图片检索，支持图搜图和文搜图
            
            ## 📊 服务状态
            - 数据服务: ✅ 运行中
            - 索引服务: ✅ 运行中
            - 模型服务: ✅ 运行中
            - 上下文工程服务: ✅ 运行中 (需要Ollama支持)
            - 图片服务: ✅ 运行中 (基于CLIP模型)
            """)
            
            with gr.Tabs():
                with gr.Tab("🏗️ 第一部分：离线索引构建"):
                    build_index_tab(self.index_service)
                with gr.Tab("🔍 第二部分：在线召回排序"):
                    build_search_tab(self.index_service, self.data_service)
                with gr.Tab("🤖 第三部分：上下文工程"):
                    build_rag_tab(self.index_service)
                with gr.Tab("🖼️ 第四部分：图片检索系统"):
                    build_image_tab(self.image_service)
                with gr.Tab("📊 第五部分：数据回收训练"):
                    build_training_tab(self.model_service, self.data_service)
                with gr.Tab("🛡️ 系统监控"):
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
            self.interface.launch(share=False, inbrowser=True, server_port=port)
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            # 尝试其他端口
            for alt_port in [7862, 7863, 7864, 7865]:
                try:
                    print(f"🔄 尝试端口 {alt_port}...")
                    self.interface.launch(share=False, inbrowser=True, server_port=alt_port)
                    break
                except Exception as e2:
                    print(f"❌ 端口 {alt_port} 也失败: {e2}")
                    continue

def main():
    ui = SearchUI()
    ui.run()

if __name__ == "__main__":
    main()
