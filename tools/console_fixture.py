"""Deterministic local callbacks for console CI. No models, desktop actions or external calls."""
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
from src.search_engine.native_runtime import NativeRuntime
from src.search_engine.workbench_runs import RunStore
from src.search_engine.workbench_api import make_handler
from src.search_engine.workbench_telemetry import record_usage

MODULES = ['离线索引构建', '在线召回排序', 'RAG', 'OAG 图谱推理', '上下文工程', '多模态系统', '数据回收训练', '系统监控']


def main():
    with tempfile.TemporaryDirectory(prefix='test-bed-console-ci-') as folder:
        report = Path(folder) / 'report.txt'
        report.write_text('可核对的运行产物\n', encoding='utf-8')
        calls = {'long': 0}
        with gr.Blocks() as blocks:
            with gr.Tabs():
                for index, module in enumerate(MODULES):
                    with gr.Tab(module):
                        if index == 1:
                            query = gr.Textbox(value='机器学习', label='实验查询')
                            rank = gr.Dropdown(['tfidf', 'ctr'], value='tfidf', label='排序算法')
                            output = gr.JSON(label='检索结果')
                            artifact = gr.File(label='运行产物')
                            def search(text, algorithm):
                                record_usage(SimpleNamespace(model='fixture', usage={'prompt_tokens': 10, 'completion_tokens': 5}), 'fixture')
                                return {'answer': text, 'documents': ['doc1'], 'algorithm': algorithm, 'score': .8}, str(report)
                            gr.Button('执行检索').click(search, [query, rank], [output, artifact], api_name='search')
                        elif index == 2:
                            text = gr.Textbox(label='问题', value='刷新验证')
                            output = gr.Textbox(label='回答')
                            def stream(text):
                                calls['long'] += 1
                                for step in range(10):
                                    yield '步骤 ' + str(step)
                                    time.sleep(.3)
                                yield f"结果：{text}，执行次数 {calls['long']}"
                            def fail():
                                raise ValueError('可恢复的验收错误')
                            gr.Button('执行长任务').click(stream, text, output, api_name='long')
                            gr.Button('触发失败').click(fail, [], output, api_name='fail')
                        else:
                            output = gr.JSON(label='模块状态')
                            gr.Button('查看状态').click(lambda: {'source': 'deterministic-ci-fixture', 'ready': True}, [], output)
        executor = NativeRuntime(lambda: blocks)
        records = RunStore(executor, folder, background=True)
        server = ThreadingHTTPServer(('127.0.0.1', 8768), make_handler(executor, records))
        try:
            server.serve_forever()
        finally:
            server.server_close()
            records.close()
            executor.close()


if __name__ == '__main__':
    main()
