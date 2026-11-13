"""Existing LLaMA-Factory parameter mapping, independent of UI/runtime state."""
def build_train_args(config):
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
