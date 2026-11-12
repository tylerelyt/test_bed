import json
import threading
import unittest
import tempfile
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import gradio as gr
from src.search_engine.native_runtime import NativeRuntime
from src.search_engine.workbench_api import make_handler
from src.search_engine.workbench_runs import RunStore

class NativeHTTPTests(unittest.TestCase):
    def setUp(self):
        with gr.Blocks() as app:
            source=gr.Textbox(value='原输入')
            output=gr.JSON()
            gr.Button('执行').click(lambda text:{'text':text},source,output,api_name='actual')
            self.upload=gr.File()
            self.file_output=gr.File()
            gr.Button('文件').click(lambda file:file,self.upload,self.file_output,api_name='file')
        self.runtime=NativeRuntime(lambda:app)
        self.addCleanup(self.runtime.close)
        self.layout=self.runtime.layout('test')
        self.source=source._id
        self.fn=next(d['id'] for d in self.layout['dependencies'] if d['api_name']=='actual')
        self.file_fn=next(d['id'] for d in self.layout['dependencies'] if d['api_name']=='file')
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.records=RunStore(self.runtime,directory.name)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.runtime,self.records))
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):self.server.shutdown();self.server.server_close()
    def test_real_http_layout_event_and_json_output(self):
        with urlopen(Request(self.base+'/api/native/layout',data=b'{"session":"test"}')) as response:
            layout=json.load(response)
        self.assertEqual(len(layout['dependencies']),2)
        payload={'session':'test','fn_id':self.fn,'values':{str(self.source):'真实HTTP输入'}}
        with urlopen(Request(self.base+'/api/native/event',data=json.dumps(payload).encode())) as response:
            events=[json.loads(line) for line in response]
        self.assertEqual(events[-1]['type'],'completed')
        self.assertIn('真实HTTP输入',json.dumps(events,ensure_ascii=False))
    def test_invalid_input_and_origin(self):
        for payload,headers,status in [({'session':'test','fn_id':-1},{},400),({'session':'test','fn_id':True},{},400),({'session':'test'},{'Origin':'https://example.com'},403)]:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.base+'/api/native/event',data=json.dumps(payload).encode(),headers=headers))
            self.assertEqual(error.exception.code,status)

    def test_streamed_upload_original_file_callback_and_download(self):
        payload=b'actual-streamed-file-'*120000
        request=Request(self.base+'/api/native/upload',data=payload,headers={'X-File-Name':'sample.bin'})
        with urlopen(request) as response:
            file=json.load(response)
        event={'session':'test','fn_id':self.file_fn,'values':{str(self.upload._id):file}}
        with urlopen(Request(self.base+'/api/native/event',data=json.dumps(event).encode())) as response:
            events=[json.loads(line) for line in response]
        self.assertEqual(events[-1]['type'],'completed')
        output=events[-1]['updates'][str(self.file_output._id)]['value']
        with urlopen(self.base+output['url']) as response:
            self.assertEqual(response.read(),payload)
            self.assertIn('sample.bin',response.headers['Content-Disposition'])

    def test_detached_run_task_restore_and_compare_over_http(self):
        def post(path,payload):
            with urlopen(Request(self.base+'/api/native/'+path,data=json.dumps(payload).encode())) as response:
                return json.load(response)
        first=post('runs',{'session':'test','fn_id':self.fn,'values':{str(self.source):'基线'}})
        # Starting a run does not require keeping the POST connection open.
        with urlopen(self.base+'/api/native/runs/'+first['run_id']+'/events') as response:
            events=[json.loads(line) for line in response]
        self.assertEqual(events[-1]['status'],'completed')
        task=post('tasks',{'action':'save','run_id':first['run_id'],'name':'固定问答','expectation':'包含原输入'})
        restored=post('tasks',{'action':'restore','task_id':task['id']})
        self.assertEqual(restored['values'][str(self.source)],'基线')
        second=post('runs',{'session':'test','fn_id':self.fn,'values':{str(self.source):'变更'},'task_id':task['id']})
        with urlopen(self.base+'/api/native/runs/'+second['run_id']+'/events?after=0') as response:
            events=[json.loads(line) for line in response]
        comparison=post('tasks',{'action':'compare','task_id':task['id'],'run_id':second['run_id']})
        self.assertEqual(comparison['inputs'][0]['after'],'变更')
        self.assertEqual(len(comparison['outputs']),1)
        with self.assertRaises(HTTPError) as error:
            post('tasks',{'action':'restore','task_id':{}})
        self.assertEqual(error.exception.code,400)

if __name__=='__main__':unittest.main()
