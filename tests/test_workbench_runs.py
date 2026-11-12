import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
import gradio as gr
from src.search_engine.native_runtime import NativeRuntime
from src.search_engine.workbench_runs import RunStore, HIDDEN


class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.release=threading.Event()
        self.started=threading.Event()
        self.calls=0
        with gr.Blocks() as blocks:
            with gr.Tab('上下文工程'):
                self.query=gr.Textbox(value='默认任务',label='问题')
                self.secret=gr.Textbox(type='password',label='API Key')
                self.upload=gr.File()
                self.output=gr.JSON(label='结果')
                self.file_output=gr.File()
                def work(query,secret):
                    self.calls+=1
                    self.started.set()
                    self.release.wait(3)
                    return {'answer':query,'echo':secret,'api_key':secret}
                def fail():
                    raise RuntimeError('明确失败')
                gr.Button('执行').click(work,[self.query,self.secret],self.output,api_name='work')
                gr.Button('失败').click(fail,[],self.output,api_name='fail')
                gr.Button('停止').click(self.release.set,[],[],api_name='stop')
                gr.Button('文件').click(lambda file:file,self.upload,self.file_output,api_name='file')
        self.runtime=NativeRuntime(lambda:blocks)
        self.addCleanup(self.runtime.close)
        self.runtime.layout('a')
        self.ids={d['api_name']:d['id'] for d in blocks.config['dependencies']}
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store=RunStore(self.runtime,self.directory.name)
        self.addCleanup(self.store.close)

    def run_work(self,text='原问题'):
        self.release.set()
        result=self.store.start('a',self.ids['work'],{str(self.query._id):text,str(self.secret._id):'private-key-value'})
        list(self.store.events(result['run_id']))
        return self.store.get('runs',result['run_id'])

    def test_persistence_redaction_baseline_and_explicit_memory(self):
        first=self.run_work()
        task=self.store.save_task(first['id'],'固定任务','答案应包含问题')
        self.store.evaluations.review(first['id'],True,'核对结果包含原问题')
        self.store.baseline(task['id'],first['id'])
        second=self.run_work('变更问题')
        comparison=self.store.compare(task['id'],second['id'])
        self.assertEqual(comparison['inputs'][0]['after'],'变更问题')
        self.assertNotEqual(first['config_version'],second['config_version'])
        self.assertEqual(second['inputs'][str(self.secret._id)],HIDDEN)
        restored=self.store.restore(task['id'])
        self.assertEqual(restored['missing'],[str(self.secret._id)])
        self.assertEqual(restored['values'][str(self.query._id)],'原问题')
        self.store.save_note(second['id'],'适用条件','已核对输入；只用于此类问题')
        reopened=RunStore(self.runtime,self.directory.name)
        self.assertEqual(reopened.get('runs',first['id'])['status'],'completed')
        self.assertEqual(reopened.catalog('上下文工程')['notes'][0]['source_run'],second['id'])
        self.assertNotIn(b'private-key-value',self.store.path.read_bytes())
        failed=self.store.start('a',self.ids['fail'],{})['run_id']
        list(self.store.events(failed))
        with self.assertRaises(ValueError):
            self.store.baseline(task['id'],failed)

    def test_task_version_is_checked_before_execution(self):
        run = self.run_work()
        task = self.store.save_task(run['id'], '版本任务', '核对输入版本')
        changed = self.store.update_task(task['id'], [], expected_version=1)
        calls = self.calls
        with self.assertRaisesRegex(ValueError, '已有新版本'):
            self.store.start('a', self.ids['work'], {}, task_id=task['id'], expected_task_version=1)
        self.assertEqual(self.calls, calls)
        result = self.store.start('a', self.ids['work'], {}, task_id=task['id'], expected_task_version=changed['version'])
        list(self.store.events(result['run_id']))
        stored = self.store.get('runs', result['run_id'])
        self.assertEqual(stored['task_version'], 2)
        self.assertEqual(stored['runtime_id'], self.runtime.instance_id)

    def test_resume_events_after_disconnect_does_not_reexecute(self):
        request_id='a'*32
        result=self.store.start('a',self.ids['work'],{str(self.query._id):'只运行一次'},request_id=request_id)
        repeated=self.store.start('a',self.ids['work'],{str(self.query._id):'只运行一次'},request_id=request_id)
        self.assertEqual(result,repeated)
        stream=self.store.events(result['run_id'])
        first=next(stream)
        stream.close()
        self.assertTrue(self.started.wait(2))
        self.release.set()
        remaining=list(self.store.events(result['run_id'],first['seq']))
        self.assertEqual(remaining[-1]['status'],'completed')
        self.assertTrue(all(e['seq']>first['seq'] for e in remaining if e['type'] not in ('settled','heartbeat')))
        self.assertEqual(self.calls,1)

    def test_interrupted_record_is_preserved_without_rerun(self):
        run=self.run_work()
        run['status']='running'
        self.store.put('runs',run)
        restarted=RunStore(self.runtime,self.directory.name)
        self.assertEqual(restarted.get('runs',run['id'])['status'],'interrupted')
        self.assertEqual(self.calls,1)
        self.assertEqual(list(restarted.events(run['id']))[-1]['status'],'interrupted')

    def test_recorded_stop_is_independent_of_long_running_callback(self):
        work=self.store.start('a',self.ids['work'],{})
        self.assertTrue(self.started.wait(2))
        stop=self.store.start('a',self.ids['stop'],{})
        self.assertEqual(list(self.store.events(stop['run_id']))[-1]['status'],'completed')
        self.assertEqual(list(self.store.events(work['run_id']))[-1]['status'],'completed')

    def test_file_live_preview_but_no_stale_file_restoration(self):
        file=self.runtime.store_upload('input.txt',io.BytesIO(b'real-bytes'),10)
        result=self.store.start('a',self.ids['file'],{str(self.upload._id):file})
        events=list(self.store.events(result['run_id']))
        output=next(e['updates'][str(self.file_output._id)]['value'] for e in events if e['type']=='update')
        self.assertEqual(self.runtime.assets[output['url'].rsplit('/',1)[-1]].read_bytes(),b'real-bytes')
        record=self.store.get('runs',result['run_id'])
        self.assertNotIn('/api/native/assets/',json.dumps(record))
        task=self.store.save_task(record['id'],'文件任务','文件字节一致')
        self.assertEqual(self.store.restore(task['id'])['missing'],[str(self.upload._id)])

    def test_failure_and_schema_incompatibility_are_explicit(self):
        result=self.store.start('a',self.ids['fail'],{})
        self.assertEqual(list(self.store.events(result['run_id']))[-1]['status'],'failed')
        record=self.store.get('runs',result['run_id'])
        self.assertEqual(record['error'],'明确失败')
        task=self.store.save_task(record['id'],'失败案例','应观察到明确失败')
        self.assertIsNone(task['baseline'])
        with self.assertRaises(ValueError):self.store.compare(task['id'],record['id'])
        task['schema']='old-version'
        self.store.put('tasks',task)
        with self.assertRaises(ValueError):self.store.restore(task['id'])
        with self.assertRaises(ValueError):self.store.start('a',self.ids['work'],{},task_id=task['id'])


if __name__=='__main__':unittest.main()
