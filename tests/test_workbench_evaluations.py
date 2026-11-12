import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
import gradio as gr
from src.search_engine.native_runtime import NativeRuntime
from src.search_engine.workbench_runs import RunStore
from src.search_engine.workbench_evaluations import grade, model_grade, validate_rules
from src.search_engine.workbench_telemetry import record_usage


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.calls = 0
        self.release = threading.Event()
        with gr.Blocks() as blocks:
            with gr.Tab('在线召回排序'):
                self.query = gr.Textbox(value='机器学习', label='问题')
                self.model = gr.Dropdown(['small', 'large'], value='small', label='模型')
                self.out = gr.JSON(label='结果')
                def answer(text, model):
                    self.calls += 1
                    record_usage(SimpleNamespace(model=model, usage={'prompt_tokens': 10, 'completion_tokens': 5}))
                    return {'answer': text, 'score': .8, 'model': model}
                def wait(text):
                    self.release.wait(4)
                    return {'text': text}
                gr.Button('查询').click(answer, [self.query, self.model], self.out, api_name='answer')
                gr.Button('长任务').click(wait, self.query, self.out, api_name='wait')
        self.runtime = NativeRuntime(lambda: blocks)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(self.runtime.close)
        self.runtime.layout('test')
        self.ids = {d['api_name']: d['id'] for d in blocks.config['dependencies']}
        self.store = RunStore(self.runtime, directory.name)
        self.addCleanup(self.store.close)
        self.addCleanup(self.release.set)
        self.eval = self.store.evaluations

    def run_answer(self, text='机器学习', task=None):
        result = self.store.start('test', self.ids['answer'], {str(self.query._id): text}, task_id=task)
        list(self.store.events(result['run_id']))
        return self.store.get('runs', result['run_id'])

    def task(self):
        run = self.run_answer()
        return self.store.save_task(run['id'], '固定问题', '回答包含问题',
            [{'op': 'contains', 'component': str(self.out._id), 'pointer': '/answer', 'expected': '机器学习'}])

    def finish_batch(self, batch):
        for cell in batch['cells']:
            list(self.store.events(cell['run_id']))
        return self.eval.summary(batch['id'])

    def test_rules_are_real_and_completion_is_not_acceptance(self):
        task = self.task()
        passed = self.run_answer(task=task['id'])
        failed = self.run_answer('无关问题', task['id'])
        self.assertEqual(passed['evaluation']['verdict'], 'passed')
        self.assertEqual(failed['status'], 'completed')
        self.assertEqual(failed['evaluation']['verdict'], 'failed')
        with self.assertRaises(ValueError): self.store.baseline(task['id'], passed['id'])
        self.eval.review(passed['id'], True, '核对文档和回答')
        self.store.baseline(task['id'], passed['id'])
        self.assertEqual(self.store.get('tasks', task['id'])['baseline'], passed['id'])

    def test_structured_checks_and_unobservable_are_separate(self):
        run = self.run_answer()
        rules = [
            {'op': 'json_keys', 'component': str(self.out._id), 'expected': ['answer', 'score']},
            {'op': 'number_between', 'component': str(self.out._id), 'pointer': '/score', 'minimum': .7, 'maximum': 1},
            {'op': 'not_contains', 'component': str(self.out._id), 'expected': '错误'},
            {'op': 'equals', 'component': str(self.out._id), 'pointer': '/model', 'expected': 'small'},
        ]
        task = {'id': 'rules', 'rules': validate_rules(rules)}
        self.assertEqual(grade(run, task)['verdict'], 'passed')
        task['rules'] = [{'op': 'equals', 'component': str(self.out._id), 'pointer': '/missing', 'expected': 1}]
        self.assertEqual(grade(run, task)['verdict'], 'blocked')
        task['rules'] = []
        self.assertEqual(grade(run, task)['verdict'], 'ungraded')
        with self.assertRaises(ValueError): validate_rules([{'op': 'number_between', 'minimum': float('nan'), 'maximum': 1}])

    def test_probe_sweep_repeats_usage_and_baseline(self):
        task = self.task()
        suite = self.eval.save_suite('在线召回排序', '模型对比', [task['id']],
            [{'name': '小模型', 'values': {str(self.model._id): 'small'}}, {'name': '大模型', 'values': {str(self.model._id): 'large'}}], 2)
        prices = {'small': {'input_per_million': 1, 'output_per_million': 2, 'source': '测试定价', 'currency': 'CNY'},
                  'large': {'input_per_million': 3, 'output_per_million': 6, 'source': '测试定价', 'currency': 'CNY'}}
        with patch.dict(os.environ, {'WORKBENCH_MODEL_PRICES': json.dumps(prices)}):
            probe = self.finish_batch(self.eval.start(suite['id'], 'probe'))
            full = self.finish_batch(self.eval.start(suite['id']))
        self.assertEqual(len(probe['rows']), 2)
        self.assertEqual(len(full['rows']), 4)
        self.assertTrue(full['complete'])
        self.assertTrue(all(group['pass_rate'] == 1 for group in full['groups']))
        self.assertEqual(full['groups'][0]['input_tokens'], 20)
        self.assertAlmostEqual(full['groups'][0]['cost'], .00004)
        self.assertGreater(full['groups'][0]['p95_seconds'], 0)
        self.assertEqual(full['recommendation']['variant'],'小模型')
        self.eval.baseline(suite['id'], full['id'])
        self.assertEqual(self.eval.compare(suite['id'], full['id'])['groups'][0]['pass_rate_delta'], 0)

    def test_missing_price_is_unknown_and_idempotency_does_not_repeat(self):
        task = self.task()
        suite = self.eval.save_suite('在线召回排序', '一次执行', [task['id']], [{'name': '当前', 'values': {}}], 1)
        first = self.finish_batch(self.eval.start(suite['id'], request_id='b' * 32))
        calls = self.calls
        repeated = self.eval.start(suite['id'], request_id='b' * 32)
        self.assertEqual(first['id'], repeated['id'])
        self.assertEqual(self.calls, calls)
        self.assertIsNone(first['groups'][0]['cost'])

    def test_suite_audit_detects_duplicate_inputs_ungraded_tasks_and_unlabelled_edges(self):
        first=self.task()
        source=self.run_answer()
        second=self.store.save_task(source['id'],'相同输入另一个任务','人工核对')
        suite=self.eval.save_suite('在线召回排序','检查题目',[first['id'],second['id']],[{'name':'默认','values':{}}],1)
        report=self.eval.audit(suite['id'])
        self.assertEqual(len(report['duplicates']),1)
        self.assertIn(second['name'],report['ungraded'])
        self.assertEqual(report['coverage']['unspecified'],2)
        self.assertTrue(any('反例' in check for check in report['checks']))

    def test_suite_parameter_versions_compare_against_fixed_tasks_and_rules(self):
        task=self.task()
        suite=self.eval.save_suite('在线召回排序','提示词迭代',[task['id']],[{'name':'当前配置','values':{}}],1)
        baseline=self.finish_batch(self.eval.start(suite['id']))
        self.eval.baseline(suite['id'],baseline['id'])
        updated=self.eval.save_suite('在线召回排序','提示词迭代',[task['id']],[{'name':'当前配置','values':{str(self.query._id):'新提示词'}}],1,suite['id'],1)
        self.assertEqual(updated['version'],2)
        changed=self.finish_batch(self.eval.start(suite['id']))
        comparison=self.eval.compare(suite['id'],changed['id'])
        self.assertEqual(comparison['groups'][0]['pass_rate_delta'],-1)
        with self.assertRaises(ValueError):self.eval.save_suite('在线召回排序','旧页面覆盖',[task['id']],[{'name':'默认','values':{}}],1,suite['id'],1)

    def test_review_gate_reject_and_queued_cancel(self):
        task = self.task()
        self.store.update_task(task['id'], task['rules'], True)
        result = self.store.start('test', self.ids['answer'], {}, task_id=task['id'])
        self.assertEqual(self.store.get('runs', result['run_id'])['status'], 'awaiting_review')
        before = self.calls
        self.store.decide(result['run_id'], 'reject')
        self.assertEqual(list(self.store.events(result['run_id']))[-1]['status'], 'rejected')
        self.assertEqual(self.calls, before)
        approved = self.store.start('test', self.ids['answer'], {}, task_id=task['id'])
        self.store.decide(approved['run_id'], 'approve')
        self.assertEqual(list(self.store.events(approved['run_id']))[-1]['status'], 'completed')
        first = self.store.start('test', self.ids['wait'], {})
        second = self.store.start('test', self.ids['wait'], {})
        self.store.decide(second['run_id'], 'cancel')
        self.release.set()
        list(self.store.events(first['run_id']))
        self.assertEqual(list(self.store.events(second['run_id']))[-1]['status'], 'cancelled')

    def test_schedule_coalesces_missed_periods_and_pauses_invalid_versions(self):
        task = self.task()
        suite = self.eval.save_suite('在线召回排序', '周期验证', [task['id']], [{'name': '默认', 'values': {}}], 1)
        schedule = self.eval.schedule(suite['id'], 5)
        now = schedule['next_at'] + 900
        self.eval.tick(now)
        current = self.store.get('schedules', schedule['id'])
        self.finish_batch(self.eval.summary(current['last_batch']))
        count = self.calls
        self.eval.tick(now)
        self.assertEqual(self.calls, count)
        self.store.update_task(task['id'], task['rules'])
        self.eval.tick(current['next_at'])
        invalid = self.store.get('schedules', schedule['id'])
        self.assertFalse(invalid['enabled'])
        self.assertIn('版本', invalid['error'])

    def test_consolidation_is_a_draft_with_multiple_sources_and_conflict_detection(self):
        first,second=self.run_answer(),self.run_answer('其他问题')
        note=self.store.consolidate('多次运行经验',[first['id'],second['id']])
        self.assertEqual(note['status'],'draft')
        self.assertEqual(len(note['source_versions']),2)
        adopted=self.store.update_note(note['id'],1,note['body'],'approved')
        self.assertEqual(adopted['version'],2)
        with self.assertRaises(ValueError):self.store.update_note(note['id'],1,note['body'],'retired')
        task=self.task()
        self.store.update_task(task['id'],task['rules'],expected_version=1)
        with self.assertRaises(ValueError):self.store.update_task(task['id'],task['rules'],expected_version=1)

    def test_schedules_survive_restart_and_operator_pause_is_not_overwritten(self):
        task=self.task()
        suite=self.eval.save_suite('在线召回排序','重启周期',[task['id']],[{'name':'默认','values':{}}],1)
        schedule=self.eval.schedule(suite['id'],5)
        restarted=RunStore(self.runtime,self.store.directory)
        self.addCleanup(restarted.close)
        restarted.evaluations.tick(schedule['next_at'])
        current=restarted.get('schedules',schedule['id'])
        for cell in restarted.evaluations.summary(current['last_batch'])['cells']:list(restarted.events(cell['run_id']))
        calls=self.calls
        restarted.evaluations.tick(schedule['next_at'])
        self.assertEqual(self.calls,calls)
        original=restarted.evaluations.start
        def pause_during_tick(*args,**kwargs):
            restarted.evaluations.schedule(suite['id'],5,False,schedule['id'])
            return original(*args,**kwargs)
        with patch.object(restarted.evaluations,'start',side_effect=pause_during_tick):
            restarted.evaluations.tick(current['next_at'])
        final=restarted.get('schedules',schedule['id'])
        self.assertFalse(final['enabled'])
        for cell in restarted.evaluations.summary(final['last_batch'])['cells']:list(restarted.events(cell['run_id']))

    def test_global_concurrency_is_bounded_and_stop_has_a_separate_lane(self):
        active=0;maximum=0;lock=threading.Lock();release=threading.Event()
        with gr.Blocks() as blocks:
            output=gr.JSON()
            def work():
                nonlocal active,maximum
                with lock:
                    active+=1;maximum=max(maximum,active)
                release.wait(4)
                with lock:active-=1
                return {'done': True}
            for index in range(4):
                with gr.Tab(f'独立模块{index}'):gr.Button(f'工作{index}').click(work,[],output)
            gr.Button('停止等待').click(release.set,[],[])
        runtime=NativeRuntime(lambda:blocks)
        self.addCleanup(runtime.close)
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'WORKBENCH_CONCURRENCY':'2'}):
            store=RunStore(runtime,directory)
            try:
                layout=runtime.layout('bounded')
                jobs=[store.start('bounded',dependency['id'],{}) for dependency in layout['dependencies'][:4]]
                deadline=time.monotonic()+2
                while active<2 and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(active,2)
                stop=store.start('bounded',layout['dependencies'][-1]['id'],{})
                while not release.is_set() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(release.is_set())
                for job in [*jobs,stop]:list(store.events(job['run_id']))
                self.assertLessEqual(maximum,2)
            finally:
                release.set();store.close()

    def test_secret_environment_reference_runs_a_batch_without_disclosing_the_key(self):
        seen=[]
        with gr.Blocks() as blocks:
            secret=gr.Textbox(type='password',label='API Key')
            out=gr.JSON()
            def authenticate(key):
                seen.append(key)
                return {'authenticated':key=='fixture-private-key','echo':key}
            gr.Button('认证检查').click(authenticate,secret,out)
        runtime=NativeRuntime(lambda:blocks)
        self.addCleanup(runtime.close)
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'TASK_API_KEY':'fixture-private-key'}):
            store=RunStore(runtime,directory)
            try:
                initial=store.start('secrets',runtime.layout('secrets')['dependencies'][0]['id'],{str(secret._id):'fixture-private-key'})
                list(store.events(initial['run_id']))
                task=store.save_task(initial['run_id'],'密钥引用任务','认证成功',[{'op':'equals','component':str(out._id),'pointer':'/authenticated','expected':True}])
                store.update_task(task['id'],task['rules'],secret_env={str(secret._id):'TASK_API_KEY'})
                restored=store.restore(task['id'])
                self.assertFalse(restored['missing'])
                self.assertNotIn(str(secret._id),restored['values'])
                suite=store.evaluations.save_suite('工作台','服务端密钥',[task['id']],[{'name':'默认','values':{}}],1)
                batch=store.evaluations.start(suite['id'])
                events=[]
                for cell in batch['cells']:events.extend(store.events(cell['run_id']))
                self.assertNotIn('fixture-private-key',json.dumps(events))
                self.assertEqual(store.evaluations.summary(batch['id'])['groups'][0]['passed'],1)
                self.assertEqual(seen,['fixture-private-key','fixture-private-key'])
                self.assertNotIn(b'fixture-private-key',store.path.read_bytes())
                child=store.get('runs',batch['cells'][0]['run_id'])
                self.assertNotIn('fixture-private-key',json.dumps(runtime.layout(child['session'])))
                with self.assertRaises(ValueError):store.update_task(task['id'],task['rules'],secret_env={str(secret._id):'HOME'})
            finally:store.close()

    def test_context_versions_are_in_the_actual_callback_input_without_repeated_prefixing(self):
        task=self.task()
        source=self.store.context.resource('验收方法','instructions','先查依据再下结论')
        bundle=self.store.context.bundle('企业配置',[{'id':source['id'],'required':True}])
        self.store.update_task(task['id'],[],context_binding={'bundle_id':bundle['id'],'version':1,'component_id':str(self.query._id)})
        result=self.run_answer(task=task['id'])
        self.assertIn('先查依据再下结论',result['outputs'][str(self.out._id)]['value']['answer'])
        self.assertEqual(result['context']['resources'][0]['version'],1)
        self.assertEqual(result['user_inputs'][str(self.query._id)],'机器学习')
        copied=self.store.save_task(result['id'],'复制配置任务','核对输入和依据')
        self.assertEqual(self.store.restore(copied['id'])['values'][str(self.query._id)],'机器学习')
        again=self.run_answer(task=copied['id'])
        self.assertEqual(again['inputs'][str(self.query._id)].count('[企业上下文]'),1)
        self.assertEqual(self.runtime.session('test')['values'][str(self.query._id)],'机器学习')
        self.store.context.resource('验收方法','instructions','新版方法',resource_id=source['id'],expected_version=1)
        pinned=self.run_answer(task=task['id'])
        self.assertIn('先查依据再下结论',pinned['inputs'][str(self.query._id)])
        self.assertNotIn('新版方法',pinned['inputs'][str(self.query._id)])

    def test_model_consolidation_stays_a_draft_and_rejects_malformed_output(self):
        first,second=self.run_answer(),self.run_answer('另一种输入')
        with patch('src.search_engine.workbench_evaluations.judge_json',return_value={'body':'基于两次记录的经验草稿，需要人工核对','status':'approved'}):
            note=self.store.consolidate('模型整理',[first['id'],second['id']],True)
        self.assertEqual(note['status'],'draft')
        self.assertEqual(note['consolidation'],'model_assisted')
        with patch('src.search_engine.workbench_evaluations.judge_json',return_value={'body':None}):
            with self.assertRaises(ValueError):self.store.consolidate('无效模型输出',[first['id'],second['id']],True)

    def test_stop_request_does_not_claim_that_an_external_action_was_cancelled(self):
        result=self.store.start('test',self.ids['wait'],{})
        deadline=time.monotonic()+2
        while self.store.get('runs',result['run_id'])['status']=='queued' and time.monotonic()<deadline:time.sleep(.01)
        self.store.decide(result['run_id'],'cancel');self.release.set()
        self.assertEqual(list(self.store.events(result['run_id']))[-1]['status'],'stopped')
        run=self.store.get('runs',result['run_id'])
        self.assertTrue(run['cancel_requested'])

    def test_scheduler_error_is_observable_without_spawning_replacement_work(self):
        with patch.object(self.eval,'tick',side_effect=OSError('数据库不可写')) as tick:
            thread=threading.Thread(target=self.eval.loop)
            thread.start()
            try:
                deadline=time.monotonic()+3
                while self.eval.last_error is None and time.monotonic()<deadline:time.sleep(.02)
                self.assertEqual(self.eval.last_error,'数据库不可写')
                self.assertIsNotNone(self.eval.last_tick)
                self.assertGreaterEqual(tick.call_count,1)
            finally:self.eval.stop.set();thread.join(2)

    def test_periodic_runs_do_not_overlap_and_interrupted_runs_pause_the_schedule(self):
        self.release.set()
        source=self.store.start('test',self.ids['wait'],{})
        list(self.store.events(source['run_id']))
        task=self.store.save_task(source['run_id'],'周期长任务','核对完成结果')
        suite=self.eval.save_suite('在线召回排序','长任务去重',[task['id']],[{'name':'默认','values':{}}],1)
        schedule=self.eval.schedule(suite['id'],1)
        self.release.clear()
        self.eval.tick(schedule['next_at'])
        first=self.store.get('schedules',schedule['id'])
        self.eval.tick(first['next_at'])
        coalesced=self.store.get('schedules',schedule['id'])
        self.assertEqual(coalesced['last_batch'],first['last_batch'])
        self.assertIn('合并',coalesced['coalesced_reason'])
        self.release.set()
        batch=self.eval.summary(first['last_batch'])
        for cell in batch['cells']:list(self.store.events(cell['run_id']))
        interrupted=self.store.get('runs',batch['cells'][0]['run_id'])
        interrupted['status']='interrupted';self.store.put('runs',interrupted)
        self.eval.tick(coalesced['next_at'])
        self.assertFalse(self.store.get('schedules',schedule['id'])['enabled'])


class ModelGraderTests(unittest.TestCase):
    def test_real_http_protocol_and_invalid_model_scores(self):
        score = {'passed': True, 'score': .9, 'reason': '输出包含可核对来源'}
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body = json.dumps({'choices': [{'message': {'content': json.dumps(score)}}]}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        with patch.dict(os.environ, {'WORKBENCH_JUDGE_URL': f'http://127.0.0.1:{server.server_port}/grade', 'WORKBENCH_JUDGE_MODEL': 'local-fixture'}):
            self.assertTrue(model_grade('材料', '核对来源')['passed'])
            score['score'] = 2
            with self.assertRaises(ValueError): model_grade('材料', '核对来源')
        self.assertEqual(received[0]['model'], 'local-fixture')
