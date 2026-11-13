"""Business sessions and their callback pools live in the single worker process."""
import os
import threading
import sys
from http.server import ThreadingHTTPServer
from ..native_runtime import NativeRuntime
from ..workbench_runs import RunStore
from ..workbench_api import make_handler


class BusinessRuntime:
    def __init__(self, directory):
        self.executor = NativeRuntime()
        self.records = RunStore(self.executor, directory, background=True)
        self.server = ThreadingHTTPServer(('127.0.0.1',int(os.environ.get('WORKBENCH_BUSINESS_PORT','8767'))),
                                         make_handler(self.executor,self.records))
        self.server.daemon_threads = True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True,name='business-transport')
        self.thread.start()

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=3)
        # Streaming callbacks observe the same cancellation flag as the stop buttons.
        for session in list(self.executor.sessions.values()):
            if self.executor.blocks:
                session['cancelled'].update(self.executor.blocks.fns)
        training=sys.modules.get('src.search_engine.training_tab.llamafactory_trainer')
        if training: training.stop_all_trainers()
        self.records.close(); self.executor.close()
