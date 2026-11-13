"""
LLaMA-Factory 训练服务
参考 LLaMA-Factory WebUI 的实现，使用 subprocess 启动独立训练进程
这样可以避免 "signal only works in main thread" 的问题
"""
import os
import json
import subprocess
import tempfile
import shutil
import signal
import threading
from typing import Dict, Any, Optional
from datetime import datetime


class LLaMAFactoryTrainer:
    """LLaMA-Factory 训练服务类（使用 subprocess 启动独立进程）"""
    
    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.training_status = {
            'running': False,
            'stage': None,
            'output_dir': None
        }
        self.config_file = None  # 临时配置文件路径
        self.managed = None
        self.log_stream = None
    
    def _build_train_args(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """构建训练参数，转换为 LLaMA-Factory 需要的格式
        
        参考 LLaMA-Factory WebUI 的 runner._parse_train_args 方法
        """
        args = {}
        
        # 基础配置
        args['stage'] = config.get('stage', 'sft')  # pt, sft, dpo, etc.
        args['model_name_or_path'] = config.get('model_name_or_path', '')
        
        # SFT/DPO阶段：从之前的checkpoint继续训练
        adapter_path = config.get('adapter_name_or_path', '')
        if adapter_path:
            args['adapter_name_or_path'] = adapter_path
        
        args['dataset'] = config.get('dataset', '')
        args['dataset_dir'] = config.get('dataset_dir', 'data/llmops')
        args['template'] = config.get('template', 'default')
        args['finetuning_type'] = config.get('finetuning_type', 'lora')
        
        # 训练参数
        args['output_dir'] = config.get('output_dir', '')
        args['overwrite_output_dir'] = config.get('overwrite_output_dir', True)
        args['do_train'] = True
        args['num_train_epochs'] = config.get('num_train_epochs', 3.0)
        args['learning_rate'] = config.get('learning_rate', 5e-5)
        args['per_device_train_batch_size'] = config.get('per_device_train_batch_size', 2)
        args['gradient_accumulation_steps'] = config.get('gradient_accumulation_steps', 8)
        args['cutoff_len'] = config.get('cutoff_len', 512)  # 改为512，避免小数据集被过滤
        args['max_grad_norm'] = config.get('max_grad_norm', 1.0)
        args['lr_scheduler_type'] = config.get('lr_scheduler_type', 'cosine')
        args['warmup_steps'] = config.get('warmup_steps', 0)
        args['logging_steps'] = config.get('logging_steps', 5)
        args['save_steps'] = config.get('save_steps', 100)
        args['save_strategy'] = 'steps'
        args['logging_strategy'] = 'steps'
        
        # 计算类型 (MacOS MPS 不支持 bf16/fp16，默认使用 fp32)
        compute_type = config.get('compute_type', 'fp32')
        if compute_type == 'bf16':
            args['bf16'] = True
            args['fp16'] = False
        elif compute_type == 'fp16':
            args['fp16'] = True
            args['bf16'] = False
        else:  # fp32 或其他
            args['fp16'] = False
            args['bf16'] = False
        
        # LoRA 配置
        if args['finetuning_type'] == 'lora':
            args['lora_rank'] = config.get('lora_rank', 8)
            args['lora_alpha'] = config.get('lora_alpha', 16)
            args['lora_dropout'] = config.get('lora_dropout', 0.05)
            args['lora_target'] = config.get('lora_target', 'all')
        
        # DPO 配置
        if args['stage'] == 'dpo':
            args['pref_beta'] = config.get('pref_beta', 0.1)
            args['pref_ftx'] = config.get('pref_ftx', 0.0)
            args['pref_loss'] = config.get('pref_loss', 'sigmoid')
        
        # 其他配置
        args['max_samples'] = config.get('max_samples', 100000)
        args['val_size'] = config.get('val_size', 0.0)
        args['plot_loss'] = True
        args['trust_remote_code'] = True
        args['overwrite_cache'] = True  # 总是重新处理数据，避免缓存问题
        args['preprocessing_num_workers'] = 1  # 单进程处理，避免并发问题
        
        if args['val_size'] > 1e-6:
            args['eval_strategy'] = 'steps'
            args['eval_steps'] = args['save_steps']
            args['per_device_eval_batch_size'] = args['per_device_train_batch_size']
        
        return args
    
    def _save_config_to_file(self, args: Dict[str, Any]) -> str:
        """将配置保存为临时 YAML 文件
        
        参考 LLaMA-Factory WebUI 的 save_cmd 方法
        """
        # 创建临时配置文件
        config_dir = tempfile.mkdtemp(prefix='llamafactory_')
        config_file = os.path.join(config_dir, 'train_config.yaml')
        
        # 转换参数格式（将 Python 类型转为 YAML 友好格式）
        yaml_args = {}
        for key, value in args.items():
            if isinstance(value, bool):
                yaml_args[key] = value
            elif isinstance(value, (int, float)):
                yaml_args[key] = value
            elif isinstance(value, str):
                yaml_args[key] = value
            elif value is None:
                continue  # 跳过 None 值
            else:
                yaml_args[key] = str(value)
        
        # 写入 YAML 文件
        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_args, f, default_flow_style=False, allow_unicode=True)
        
        self.config_file = config_file
        return config_file
    
    def start_training(self, config: Dict[str, Any], log_callback: Optional[callable] = None) -> bool:
        """启动训练任务（使用 subprocess 启动独立进程）
        
        参考 LLaMA-Factory WebUI 的 _launch 方法
        
        Args:
            config: 训练配置字典
            log_callback: 日志回调函数（暂不使用，保留接口兼容性）
        
        Returns:
            是否成功启动训练进程
        """
        if self.training_status['running']:
            return False
        
        # 查找 llamafactory-cli 命令
        # 优先使用系统 PATH，如果找不到则尝试 Python 环境的 bin 目录
        llamafactory_cmd = os.environ.get('WORKBENCH_TRAIN_PYTHON') or shutil.which('llamafactory-cli')
        if not llamafactory_cmd:
            # 尝试从 Python 环境的 bin 目录查找
            import sys
            python_bin_dir = os.path.dirname(sys.executable)
            llamafactory_cmd = os.path.join(python_bin_dir, 'llamafactory-cli')
            if not os.path.exists(llamafactory_cmd):
                if log_callback:
                    log_callback("❌ llamafactory-cli 命令不可用，请确保已安装 LLaMA-Factory")
                print(f"llamafactory-cli not found in PATH or {python_bin_dir}")
                return False
        
        # 构建训练参数
        train_args = self._build_train_args(config)
        from ..business_training import prepare_training
        self.managed = prepare_training(config,train_args)
        
        # 确保输出目录存在
        output_dir = train_args.get('output_dir')
        if not output_dir:
            if log_callback:
                log_callback("❌ 输出目录未指定")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存配置到临时文件
        try:
            config_file = self._save_config_to_file(train_args)
            if self.managed:
                config_file=str(self.managed["folder"]/"train-config.json")
            print(f"📄 配置文件已生成: {config_file}")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ 保存配置文件失败: {str(e)}")
            print(f"配置文件生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # 设置环境变量（参考 LLaMA-Factory WebUI）
        env = os.environ.copy()
        env['LLAMABOARD_ENABLED'] = '1'
        env['LLAMABOARD_WORKDIR'] = output_dir
        
        # 启动训练进程（使用 subprocess.Popen）
        # 参考 LLaMA-Factory WebUI: self.trainer = Popen(["llamafactory-cli", "train", save_cmd(args)], env=env, stderr=PIPE, text=True)
        try:
            print(f"🚀 启动训练进程: {llamafactory_cmd} train {config_file}")
            # 不捕获 stdout/stderr，让它直接输出到终端
            # 这样用户可以实时看到训练进度
            if self.managed:self.log_stream=(self.managed["folder"]/"process.log").open("ab")
            self.current_process = subprocess.Popen(
                [os.environ.get('WORKBENCH_TRAIN_PYTHON') or __import__('sys').executable, '-c', 'from llamafactory.cli import main; main()', 'train', config_file],
                env=env,
                start_new_session=True,
                stdout=self.log_stream, stderr=subprocess.STDOUT,
                # 注释掉管道，让输出直接显示
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                text=True
            )
            
            if self.managed:
                self.managed['scope']['run']['training']['pid']=self.current_process.pid
                self.managed['scope']['records'].put('runs',self.managed['scope']['run'])
            # 更新状态
            self.training_status = {
                'running': True,
                'stage': train_args.get('stage'),
                'output_dir': output_dir
            }
            
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ 启动训练进程失败: {str(e)}")
            print(f"训练启动异常: {str(e)}")
            import traceback
            traceback.print_exc()
            # 清理临时配置文件
            if not self.managed and self.config_file and os.path.exists(self.config_file) and 'business-processes' not in self.config_file:
                try:
                    config_dir = os.path.dirname(self.config_file)
                    shutil.rmtree(config_dir)
                except:
                    pass
            return False
    
    def get_training_logs(self, max_lines: int = 100) -> str:
        """获取训练日志（从 trainer_log.jsonl 文件）
        
        参考 LLaMA-Factory WebUI 的 get_trainer_info 方法
        """
        output_dir = self.training_status.get('output_dir')
        if not output_dir or not os.path.exists(output_dir):
            return "暂无训练日志"
        
        trainer_log_path = os.path.join(output_dir, "trainer_log.jsonl")
        if not os.path.exists(trainer_log_path):
            return "训练尚未开始或日志文件不存在"
        
        try:
            logs = []
            with open(trainer_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
            
            if not logs:
                return "暂无训练日志"
            
            # 格式化日志输出（只显示关键信息）
            latest_log = logs[-1]
            
            current_steps = latest_log.get('current_steps', 0)
            total_steps = latest_log.get('total_steps', 0)
            elapsed = latest_log.get('elapsed_time', '0:00:00')
            remaining = latest_log.get('remaining_time', '0:00:00')
            percentage = latest_log.get('percentage', 0)
            
            log_text = f"""
<div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <p><strong>📊 训练进度:</strong> {current_steps}/{total_steps} 步 ({percentage:.1f}%)</p>
    <p><strong>⏱️ 已用时间:</strong> {elapsed}</p>
    <p><strong>⏳ 剩余时间:</strong> {remaining}</p>
"""
            
            if 'loss' in latest_log:
                log_text += f'    <p><strong>📉 当前损失:</strong> {latest_log["loss"]:.4f}</p>\n'
            
            if 'learning_rate' in latest_log:
                log_text += f'    <p><strong>📈 学习率:</strong> {latest_log["learning_rate"]:.2e}</p>\n'
            
            log_text += "</div>"
            
            return log_text
        except Exception as e:
            return f"读取日志失败: {str(e)}"
    
    def get_training_progress(self) -> tuple[float, str]:
        """获取训练进度百分比和状态信息
        
        Returns:
            (进度百分比 0-100, 状态文本)
        """
        output_dir = self.training_status.get('output_dir')
        if not output_dir or not os.path.exists(output_dir):
            return 0.0, "未开始"
        
        trainer_log_path = os.path.join(output_dir, "trainer_log.jsonl")
        if not os.path.exists(trainer_log_path):
            return 0.0, "训练初始化中..."
        
        try:
            logs = []
            with open(trainer_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
            
            if not logs:
                return 0.0, "训练初始化中..."
            
            # 获取最新日志
            latest_log = logs[-1]
            current_steps = latest_log.get('current_steps', 0)
            total_steps = latest_log.get('total_steps', 1)
            
            # 计算进度百分比
            progress = (current_steps / total_steps * 100) if total_steps > 0 else 0.0
            
            # 构建状态文本
            status = f"训练中: {current_steps}/{total_steps} 步"
            if 'loss' in latest_log:
                status += f" | 损失: {latest_log['loss']:.4f}"
            
            return progress, status
            
        except Exception as e:
            return 0.0, f"读取进度失败: {str(e)}"
    
    def check_process_status(self) -> Optional[int]:
        """检查训练进程状态
        
        Returns:
            进程返回码（None 表示仍在运行，0 表示成功完成，其他值表示错误）
        """
        if self.current_process is None:
            return None
        
        # 非阻塞检查进程状态
        return_code = self.current_process.poll()
        
        if return_code is not None and self.training_status['running']:
            from ..business_training import training_finished
            training_finished(self.managed,return_code)
            self.managed=None
            if self.log_stream:self.log_stream.close();self.log_stream=None
            # 进程已结束，读取输出
            try:
                stdout, stderr = self.current_process.communicate(timeout=1)
                if stderr:
                    print(f"⚠️  训练进程stderr输出:\n{stderr}")
                if stdout:
                    print(f"📄 训练进程stdout输出:\n{stdout}")
            except:
                pass
            
            # 进程已结束
            self.training_status['running'] = False
            
            # 清理临时配置文件
            if not self.managed and self.config_file and os.path.exists(self.config_file) and 'business-processes' not in self.config_file:
                try:
                    config_dir = os.path.dirname(self.config_file)
                    shutil.rmtree(config_dir)
                except:
                    pass
                self.config_file = None
        
        return return_code
    
    def stop_training(self) -> bool:
        """停止训练任务（终止进程）"""
        if not self.training_status['running'] or self.current_process is None:
            return False
        
        try:
            # 优雅地终止进程
            os.killpg(self.current_process.pid,signal.SIGTERM)
            
            # 等待进程结束（最多 5 秒）
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 如果进程没有响应，强制杀死
                os.killpg(self.current_process.pid,signal.SIGKILL)
                self.current_process.wait()
            
            from ..business_training import training_finished
            training_finished(self.managed,-15)
            self.managed=None
            if self.log_stream:self.log_stream.close();self.log_stream=None
            # 更新状态
            self.training_status['running'] = False
            self.current_process = None
            
            # 清理临时配置文件
            if not self.managed and self.config_file and os.path.exists(self.config_file) and 'business-processes' not in self.config_file:
                try:
                    config_dir = os.path.dirname(self.config_file)
                    shutil.rmtree(config_dir)
                except:
                    pass
                self.config_file = None
            
            return True
            
        except Exception as e:
            print(f"停止训练进程失败: {str(e)}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return self.training_status.copy()
    
    def is_training(self) -> bool:
        """检查是否正在训练"""
        # 如果标记为运行中，再检查进程是否真的在运行
        if self.training_status['running']:
            return_code = self.check_process_status()
            return return_code is None
        return False


# One native training process per project; stop callbacks resolve the same owner.
_trainers = {}
_trainer_lock = threading.RLock()

def get_trainer() -> LLaMAFactoryTrainer:
    from ..workbench_access import identity
    with _trainer_lock:
        return _trainers.setdefault(identity().project,LLaMAFactoryTrainer())

def stop_all_trainers():
    with _trainer_lock:
        for trainer in _trainers.values():
            if trainer.training_status['running']:trainer.stop_training()

def stop_training_for_run(run_id):
    with _trainer_lock:
        for trainer in _trainers.values():
            if trainer.managed and trainer.managed['scope']['run']['id']==run_id:
                trainer.stop_training()
