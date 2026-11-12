import unittest
import base64
import tempfile
from pathlib import Path
import gradio as gr
from src.search_engine.native_runtime import NativeRuntime

class NativeRuntimeTests(unittest.TestCase):
    def create_runtime(self):
        with gr.Blocks() as app:
            with gr.Tab('原模块'):
                count=gr.State(0)
                query=gr.Textbox(value='初始值')
                choice=gr.Dropdown([1,2],value=1)
                table=gr.Dataframe(headers=['文档'],value=[['甲'],['乙']])
                result=gr.Textbox()
                output=gr.File()
                upload=gr.File()
                def increment(value,n):return value+str(n+1),n+1
                gr.Button('执行').click(increment,[query,count],[result,count],api_name='increment')
                gr.Button('刷新').click(lambda:gr.update(choices=[2,3],value=3),[],choice,api_name='refresh')
                def stream():
                    yield '步骤一'
                    yield '步骤二'
                gr.Button('流式').click(stream,[],result,api_name='stream')
                def select(data,evt:gr.SelectData):return str(evt.index[0])+evt.value
                table.select(select,table,result,api_name='select')
                gr.Button('文件').click(lambda f:f,upload,output,api_name='file')
        runtime=NativeRuntime(lambda:app)
        self.addCleanup(runtime.close)
        layout=runtime.layout('a')
        names={d['api_name']:d['id'] for d in layout['dependencies']}
        return runtime,layout,names,query,result,choice,table,upload,output

    def test_original_state_updates_stream_and_selection(self):
        runtime,layout,names,query,result,choice,table,upload,output=self.create_runtime()
        runtime.execute('a',names['increment'],{str(query._id):'任务'})
        self.assertEqual(runtime.session('a')['values'][str(result._id)],'任务1')
        runtime.execute('a',names['increment'],{})
        self.assertEqual(runtime.session('a')['values'][str(result._id)],'任务2')
        runtime.execute('b',names['increment'],{str(query._id):'任务'})
        self.assertEqual(runtime.session('b')['values'][str(result._id)],'任务1')
        runtime.execute('a',names['refresh'],{})
        self.assertEqual(runtime.session('a')['values'][str(choice._id)],3)
        events=[]
        runtime.execute('a',names['stream'],{},emit=events.append)
        updates=[e['updates'][str(result._id)]['value'] for e in events if e['type']=='update']
        self.assertIn('步骤一',updates);self.assertIn('步骤二',updates)
        runtime.execute('a',names['select'],{},event={'index':[1,0],'value':'乙','selected':True})
        self.assertEqual(runtime.session('a')['values'][str(result._id)],'1乙')

    def test_file_is_real_cached_output(self):
        runtime,layout,names,query,result,choice,table,upload,output=self.create_runtime()
        response=runtime.execute('a',names['file'],{str(upload._id):{'__upload':{'name':'sample.txt','base64':base64.b64encode(b'actual-file').decode()}}})
        value=response['updates'][str(output._id)]['value']
        token=value['path'].rsplit('/',1)[-1]
        self.assertEqual(runtime.assets[token].read_bytes(),b'actual-file')

    def test_stop_callback_can_interrupt_streaming_work(self):
        import threading,time
        flag={'stop':False}
        with gr.Blocks() as app:
            out=gr.Textbox()
            def work():
                for n in range(50):
                    if flag['stop']:
                        yield 'stopped'
                        return
                    yield str(n)
                    time.sleep(.02)
            def stop():
                flag['stop']=True
                return 'requested'
            gr.Button('开始').click(work,[],out,api_name='work')
            gr.Button('停止').click(stop,[],[],api_name='stop')
        runtime=NativeRuntime(lambda:app)
        self.addCleanup(runtime.close)
        layout=runtime.layout('a')
        ids={d['api_name']:d['id'] for d in layout['dependencies']}
        started=threading.Event()
        thread=threading.Thread(target=lambda:runtime.execute('a',ids['work'],{},emit=lambda e:started.set() if e['type']=='update' else None))
        thread.start();self.assertTrue(started.wait(2))
        runtime.execute('a',ids['stop'],{})
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(runtime.session('a')['values'][str(out._id)],'stopped')

    def test_json_path_is_not_promoted_to_a_downloadable_server_file(self):
        with tempfile.TemporaryDirectory() as directory:
            private=Path(directory)/'private.txt';private.write_text('not-a-file-output')
            with gr.Blocks() as app:
                result=gr.JSON()
                gr.Button('JSON').click(lambda:{'path':str(private)},[],result)
            runtime=NativeRuntime(lambda:app)
            self.addCleanup(runtime.close)
            layout=runtime.layout('paths')
            returned=runtime.execute('paths',layout['dependencies'][0]['id'],{})
            self.assertEqual(returned['updates'][str(result._id)]['value']['path'],str(private))
            self.assertNotIn(str(private),runtime.asset_tokens)

if __name__=='__main__':unittest.main()
