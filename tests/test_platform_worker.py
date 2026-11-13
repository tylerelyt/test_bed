"""Exercise real process ownership and signal shutdown, independently of fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class WorkerProcessTests(unittest.TestCase):
    def test_idle_sigterm_exits_and_duplicate_worker_never_claims_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'database-url.txt').write_text('sqlite:///'+str(directory/'platform.sqlite3'))
            env = {**os.environ, 'WORKBENCH_STATE_DIR':folder}
            env.pop('WORKBENCH_DATABASE_URL', None)
            marker = directory/'agent-worker.json'
            with (directory/'worker.log').open('wb') as log:
                for _ in range(3):
                    worker = subprocess.Popen([sys.executable,'-m','src.search_engine.platform_worker'],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        for _ in range(100):
                            if marker.exists() and json.loads(marker.read_text())['pid']==worker.pid:break
                            if worker.poll() is not None:break
                            time.sleep(.05)
                        self.assertIsNone(worker.poll())
                        self.assertEqual(json.loads(marker.read_text())['pid'],worker.pid)
                        duplicate = subprocess.run([sys.executable,'-m','src.search_engine.platform_worker'],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
                        self.assertNotEqual(duplicate.returncode,0)
                        self.assertEqual(json.loads(marker.read_text())['pid'],worker.pid)
                        time.sleep(.1)
                        worker.terminate()
                        self.assertEqual(worker.wait(timeout=10),0)
                        self.assertFalse(marker.exists())
                    finally:
                        if worker.poll() is None:worker.kill();worker.wait(timeout=5)


if __name__=='__main__':unittest.main()
