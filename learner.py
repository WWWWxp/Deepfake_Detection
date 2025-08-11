
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from scipy.io.wavfile import write
from sklearn.metrics import f1_score, roc_curve
import signal
import atexit

from tqdm import tqdm
import logging
import time

from dataset import from_train_list
from model import Wav2Vec2_AASIST
from params import params
from metrics import calculate_metrics_for_train, compute_class_losses, Metrics_batch, Recorder



def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


def summarize_model(net):
  total = sum(p.numel() for p in net.parameters())
  trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
  param_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
  buffer_bytes = sum(b.numel() * b.element_size() for b in net.buffers())
  size_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

  print(f"[Model Summary]")
  print(f"  • Total parameters    : {total:,}  ({total/1e6:.2f} M)")
  print(f"  • Trainable parameters: {trainable:,}  ({trainable/1e6:.2f} M)")
  print(f"  • Model size (MB)     : {size_mb:.2f}")



class Learner:
  def __init__(self, model_dir, model, dataset, dataset_unsupervised, optimizer, params, dev_dataset=None, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = os.path.join(model_dir,'models')
    self.log_file = os.path.join(model_dir,'train_log')
    self.model = model
    summarize_model(self.model)
    self.dataset = dataset
    self.dataset_unsupervised = dataset_unsupervised
    self.dev_dataset = dev_dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.amp.autocast('cuda', enabled=kwargs.get('fp16', False))
    self.scaler = torch.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.current_epoch = 0
    self.is_master = True
    self.summary_writer = None
    os.makedirs(self.model_dir, exist_ok=True)
    
    # 初始化指标记录器
    self.loss_func = nn.CrossEntropyLoss()
    self.train_metrics = Metrics_batch()
    self.loss_recorder = Recorder()
    self.real_loss_recorder = Recorder()
    self.fake_loss_recorder = Recorder()
    
    # 训练统计信息
    self.total_steps_per_epoch = len(dataset) if dataset else 0
    self.total_epochs = getattr(params, 'epochs', 10)
    
    # 注册清理函数
    atexit.register(self.cleanup)
    signal.signal(signal.SIGINT, self.signal_handler)
    signal.signal(signal.SIGTERM, self.signal_handler)

  def cleanup(self):
    """清理资源"""
    if self.summary_writer:
      self.summary_writer.close()
    torch.cuda.empty_cache()

  def signal_handler(self, signum, frame):
    """信号处理器"""
    print(f"Received signal {signum}, cleaning up...")
    self.cleanup()
    sys.exit(0)

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'epoch': getattr(self, 'current_epoch', 0),
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict, strict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'], strict=strict)
    else:
      self.model.load_state_dict(state_dict['model'], strict=strict)

    optimizer_state_dict = state_dict['optimizer']
    current_state_dict = self.optimizer.state_dict()
    for group in optimizer_state_dict['param_groups']:
      for current_group in current_state_dict['param_groups']:
        if group['params'] == current_group['params']:
          current_group.update(group)
        else: 
          print(group)
    self.optimizer.load_state_dict(current_state_dict)
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']
    self.current_epoch = state_dict.get('epoch', 0)

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def list_all_checkpoints(self):
    """列出所有可用的checkpoint"""
    if not os.path.exists(self.model_dir):
      print("No model directory found")
      return []
    
    checkpoint_files = []
    for file in os.listdir(self.model_dir):
      if file.endswith('.pt') and not os.path.islink(os.path.join(self.model_dir, file)):
        checkpoint_files.append(file)
    
    if checkpoint_files:
      print("Available checkpoints:")
      for file in sorted(checkpoint_files):
        file_path = os.path.join(self.model_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        mod_time = time.ctime(os.path.getmtime(file_path))
        print(f"  - {file} ({file_size:.1f}MB, {mod_time})")
    else:
      print("No checkpoints found")
    
    return checkpoint_files

  def find_latest_checkpoint(self):
    """查找最新的checkpoint文件，优先按epoch数字排序"""
    if not os.path.exists(self.model_dir):
      return None
    
    # 查找所有checkpoint文件
    checkpoint_files = []
    epoch_checkpoints = []
    other_checkpoints = []
    
    for file in os.listdir(self.model_dir):
      if file.endswith('.pt') and not os.path.islink(os.path.join(self.model_dir, file)):
        checkpoint_files.append(file)
        
        # 检查是否是epoch格式的checkpoint
        if file.startswith('epoch_') and file.endswith('.pt'):
          try:
            # 提取epoch数字，格式如: epoch_5-12345.pt
            parts = file.replace('.pt', '').split('-')
            epoch_part = parts[0]  # epoch_5
            epoch_num = int(epoch_part.split('_')[1])  # 5
            step_num = int(parts[1]) if len(parts) > 1 else 0  # 12345
            epoch_checkpoints.append((epoch_num, step_num, file))
          except (ValueError, IndexError):
            other_checkpoints.append(file)
        else:
          other_checkpoints.append(file)
    
    if not checkpoint_files:
      return None
    
    # 优先选择最新的epoch checkpoint
    if epoch_checkpoints:
      # 按epoch数字排序，然后按step数字排序
      epoch_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
      latest_file = epoch_checkpoints[0][2]
      latest_epoch = epoch_checkpoints[0][0]
      latest_step = epoch_checkpoints[0][1]
      print(f"Found latest epoch checkpoint: {latest_file} (Epoch {latest_epoch}, Step {latest_step})")
    else:
      # 如果没有epoch checkpoint，按修改时间排序
      other_checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)), reverse=True)
      latest_file = other_checkpoints[0]
      print(f"Found latest checkpoint: {latest_file}")
    
    return latest_file.replace('.pt', '')

  def restore_from_checkpoint(self, filename=None):
    """恢复checkpoint，如果filename为None则自动查找最新的"""
    if filename is None:
      filename = self.find_latest_checkpoint()
      if filename is None:
        print("No checkpoint found, starting from scratch")
        return False
    
    checkpoint_path = f'{self.model_dir}/{filename}.pt'
    try:
      print(f"Loading checkpoint from: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, weights_only=True)
      self.load_state_dict(checkpoint, strict=False)
      
      # 打印恢复信息
      restored_epoch = getattr(self, 'current_epoch', 0)
      restored_step = getattr(self, 'step', 0)
      print(f"Successfully restored from checkpoint:")
      print(f"  - Epoch: {restored_epoch}")
      print(f"  - Step: {restored_step}")
      
      return True
    except FileNotFoundError:
      print(f"Checkpoint file not found: {checkpoint_path}")
      return False
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    # config logging
    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)s %(levelname)s:%(message)s',
            filename=self.log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("params:{}".format(str(self.params)))
    
    # 从params读取epochs设置
    max_epochs = getattr(self.params, 'epochs', 10)
    
    # 确定开始的epoch（如果从checkpoint恢复）
    start_epoch = self.current_epoch
    if start_epoch > 0:
      print(f"Resuming training from epoch {start_epoch + 1}/{max_epochs}")
      logger.info(f"Resuming training from epoch {start_epoch + 1}/{max_epochs}")
      if start_epoch >= max_epochs:
        print(f"Training already completed! Current epoch {start_epoch} >= max epochs {max_epochs}")
        return
    else:
      print(f"Starting training for {max_epochs} epochs...")
      logger.info(f"Training for {max_epochs} epochs")
    
    start = time.time()
    
    for epoch in range(start_epoch, max_epochs):
      self.current_epoch = epoch + 1
      print(f"\n=== Starting Epoch {self.current_epoch}/{max_epochs} ===")
      print(f"Dataset size: {len(self.dataset)} batches")
      logger.info(f"Starting Epoch {self.current_epoch}/{max_epochs}")
      
      epoch_start_time = time.time()
      
      for features in tqdm(self.dataset, desc=f'Epoch {self.current_epoch}/{max_epochs}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

        loss  = self.train_step(features, self.step)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 10 == 0 or self.step % len(self.dataset) == 0:
            # 获取当前batch的指标
            current_metrics = self._get_current_metrics()
            self._write_summary(self.step, features, loss, current_metrics)
            
            # 打印详细指标
            metrics_str = f"step:{self.step}, loss:{loss:.4f}"
            if current_metrics:
              metrics_str += f", acc:{current_metrics.get('acc', 0):.4f}"
              metrics_str += f", auc:{current_metrics.get('auc', 0):.4f}"
              metrics_str += f", eer:{current_metrics.get('eer', 0):.4f}"
              metrics_str += f", real_loss:{current_metrics.get('real_loss', 0):.4f}"
              metrics_str += f", fake_loss:{current_metrics.get('fake_loss', 0):.4f}"
            
            print(f'-----{metrics_str}-----')
            logger.info(metrics_str)
            start = time.time()
            
          
          # 每100步记录平均指标
          if self.step % 100 == 0 and self.step > 0:
            avg_metrics = self.train_metrics.get_mean_metrics()
            avg_loss = self.loss_recorder.average()
            avg_real_loss = self.real_loss_recorder.average()
            avg_fake_loss = self.fake_loss_recorder.average()
            
            print(f'-----Average Metrics (last 100 steps)-----')
            print(f'Avg Loss: {avg_loss:.4f}, Real Loss: {avg_real_loss:.4f}, Fake Loss: {avg_fake_loss:.4f}')
            print(f'Avg ACC: {avg_metrics["acc"]:.4f}, AUC: {avg_metrics["auc"]:.4f}, EER: {avg_metrics["eer"]:.4f}, AP: {avg_metrics["ap"]:.4f}')
            
            # 记录到tensorboard
            if self.summary_writer:
              self.summary_writer.add_scalar('train_avg/loss', avg_loss, self.step)
              self.summary_writer.add_scalar('train_avg/real_loss', avg_real_loss, self.step)
              self.summary_writer.add_scalar('train_avg/fake_loss', avg_fake_loss, self.step)
              self.summary_writer.add_scalar('train_avg/acc', avg_metrics["acc"], self.step)
              self.summary_writer.add_scalar('train_avg/auc', avg_metrics["auc"], self.step)
              self.summary_writer.add_scalar('train_avg/eer', avg_metrics["eer"], self.step)
              self.summary_writer.add_scalar('train_avg/ap', avg_metrics["ap"], self.step)
              self.summary_writer.flush()
            
            # 清空记录器
            self.train_metrics.clear()
            self.loss_recorder.clear()
            self.real_loss_recorder.clear()
            self.fake_loss_recorder.clear()
          
          # 每1000步进行一次dev评估
          if self.step % 1000 == 0 and self.step > 0:
            print(f"Running dev evaluation at step {self.step}...")
            eer, f1 = self.evaluate_dev()
            if eer is not None and f1 is not None:
              print(f'-----Dev Results at step {self.step}: EER={eer:.4f}, F1={f1:.4f}-----')
              logger.info(f'Dev evaluation at step {self.step}: EER={eer:.4f}, F1={f1:.4f}')
              
              # 记录到tensorboard
              if self.summary_writer:
                self.summary_writer.add_scalar('dev/eer', eer, self.step)
                self.summary_writer.add_scalar('dev/f1', f1, self.step)
                self.summary_writer.flush()
          
        self.step += 1
      
      # 每个epoch结束后的操作
      epoch_time = time.time() - epoch_start_time
      steps_this_epoch = self.step - (epoch * self.total_steps_per_epoch)
      
      print(f"\n=== Epoch {self.current_epoch}/{max_epochs} Summary ===")
      print(f"Time: {epoch_time/60:.2f} minutes")
      print(f"Steps: {steps_this_epoch}")
      print(f"Total steps so far: {self.step}")
      
      logger.info(f"Epoch {self.current_epoch}/{max_epochs} completed in {epoch_time/60:.2f} minutes, {steps_this_epoch} steps")
      
      # 每个epoch结束后保存checkpoint
      if self.is_master:
        try:
          self.save_to_checkpoint(f'epoch_{self.current_epoch}')
          print(f"Checkpoint saved for epoch {self.current_epoch}")
        except Exception as e:
          print(f"Failed to save checkpoint for epoch {self.current_epoch}: {e}")
      
      # 每个epoch结束后进行dev评估
      if self.is_master and self.dev_dataset is not None:
        print(f"Running dev evaluation after epoch {self.current_epoch}...")
        eer, f1 = self.evaluate_dev()
        if eer is not None and f1 is not None:
          print(f'=== Epoch {self.current_epoch} Dev Results: EER={eer:.4f}, F1={f1:.4f} ===')
          logger.info(f'Epoch {self.current_epoch} Dev Results: EER={eer:.4f}, F1={f1:.4f}')
          
          # 记录到tensorboard
          if self.summary_writer:
            self.summary_writer.add_scalar('epoch_dev/eer', eer, self.current_epoch)
            self.summary_writer.add_scalar('epoch_dev/f1', f1, self.current_epoch)
            self.summary_writer.add_scalar('epoch_info/epoch_time_minutes', epoch_time/60, self.current_epoch)
            self.summary_writer.flush()
    
    print(f"\n=== Training completed after {max_epochs} epochs ===")
    logger.info(f"Training completed after {max_epochs} epochs")

  def train_step(self, features, step):
    self.model.train()
    for param in self.model.parameters():
        param.grad = None

    # Use waveforms instead of mels, as the model expects raw audio
    waveforms = features['waveforms']
    device = waveforms.device
    labels = features['labels']
    
    with self.autocast:
      logits = self.model(waveforms)
      
      # 计算分类别损失
      loss_dict = compute_class_losses(logits, labels, self.loss_func)
      loss = loss_dict['overall']
    
    # 计算训练指标
    with torch.no_grad():
      auc, eer, acc, ap = calculate_metrics_for_train(labels, logits)
      
      # 更新指标记录器
      if auc is not None and eer is not None:
        self.train_metrics.update(labels, logits)
      
      # 记录损失
      self.loss_recorder.update(loss.item())
      self.real_loss_recorder.update(loss_dict['real_loss'].item())
      self.fake_loss_recorder.update(loss_dict['fake_loss'].item())
      
      # 存储当前batch的指标用于显示
      self.current_batch_metrics = {
        'acc': acc,
        'auc': auc if auc is not None else 0,
        'eer': eer if eer is not None else 0,
        'ap': ap,
        'real_loss': loss_dict['real_loss'].item(),
        'fake_loss': loss_dict['fake_loss'].item()
      }

    if self.is_master and step % 50 == 0:        # 每 50 步统计一次
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:,1]           # spoof 概率 (因为标签已调换)
            mean  = probs.mean().item()
            std   = probs.std(unbiased=False).item()
            if self.summary_writer:
                self.summary_writer.add_scalar('train/prob_mean', mean, step)
                self.summary_writer.add_scalar('train/prob_std',  std,  step)
                # 也可以画直方图
                self.summary_writer.add_histogram('train/prob_hist', probs, step)
            else:
                print(f'[step {step}] P(spoof) mean={mean:.3f}  std={std:.3f}')
                
    if torch.isnan(loss) or torch.isinf(loss):
      print("Loss is NaN or Inf")
      print(f"loss: {loss}")

    assert not torch.isnan(loss), "NaN detected in loss!"
    assert logits.shape[0] == labels.shape[0], f"Batch mismatch: logits {logits.shape}, labels {labels.shape}"
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    
    return loss

  def _get_current_metrics(self):
    """获取当前batch的指标"""
    return getattr(self, 'current_batch_metrics', None)

  def _write_summary(self, step, features, loss, metrics=None):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar('train/loss', loss, step)
    
    # 记录当前batch的指标
    if metrics:
      writer.add_scalar('train_batch/acc', metrics.get('acc', 0), step)
      writer.add_scalar('train_batch/auc', metrics.get('auc', 0), step)
      writer.add_scalar('train_batch/eer', metrics.get('eer', 0), step)
      writer.add_scalar('train_batch/ap', metrics.get('ap', 0), step)
      writer.add_scalar('train_batch/real_loss', metrics.get('real_loss', 0), step)
      writer.add_scalar('train_batch/fake_loss', metrics.get('fake_loss', 0), step)
    
    writer.flush()
    self.summary_writer = writer

  def predict(self, features):
    with torch.no_grad():
      waveforms = features['waveforms']
      logits = self.model(waveforms)
      batch_out = F.softmax(logits, dim=1)
      batch_score = batch_out[:, 1].cpu().numpy().ravel()  # 提取 positive 类的概率
    return batch_score

  def calculate_eer(self, y_true, y_scores):
    """计算等错误率 (Equal Error Rate)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

  def evaluate_dev(self):
    """在开发集上进行评估，计算EER和F1分数"""
    if self.dev_dataset is None:
      print("No dev dataset provided, skipping evaluation")
      return None, None
    
    self.model.eval()
    all_scores = []
    all_labels = []
    
    device = next(self.model.parameters()).device
    
    with torch.no_grad():
      for features in tqdm(self.dev_dataset, desc="Evaluating on dev set"):
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        
        waveforms = features['waveforms']
        labels = features['labels']
        
        logits = self.model(waveforms)
        probs = F.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()  # positive class probability
        
        all_scores.extend(scores)
        all_labels.extend(labels.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 计算EER
    eer, eer_threshold = self.calculate_eer(all_labels, all_scores)
    
    # 使用EER阈值计算F1分数
    predictions = (all_scores >= eer_threshold).astype(int)
    f1 = f1_score(all_labels, predictions)
    
    return eer, f1




def _train_impl(replica_id, model, dataset, dataset_unsupervised, args, params, dev_dataset=None):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
  learner = Learner(args.model_dir, model, dataset, dataset_unsupervised, opt, params, dev_dataset=dev_dataset, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  
  # 自动恢复最新的checkpoint
  if learner.is_master:
    learner.restore_from_checkpoint()  # 自动查找最新checkpoint
  
  # 使用max_steps或epochs，优先使用max_steps
  max_steps = getattr(args, 'max_steps', None)
  learner.train(max_steps=max_steps)


def train_distributed_torchrun(replica_id, args, params):
  dataset = from_train_list(args.train_list[0], args.audio_root, params, is_distributed=True)
  
  # 创建dev数据集（如果提供了dev_list）
  dev_dataset = None
  if hasattr(args, 'dev_list') and args.dev_list:
    dev_dataset = from_train_list(args.dev_list, args.audio_root, params, is_distributed=False)
  
  # 自动获取全局设备信息
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  
  # 初始化模型
  model = Wav2Vec2_AASIST(layers=params.layers).to(device)
  
  # 初始化DDP
  model = DistributedDataParallel(
    model,
    device_ids=[replica_id],
    output_device=replica_id,
    find_unused_parameters=True  # 根据实际情况调整
  )
  _train_impl(replica_id, model, dataset, None, args, params, dev_dataset=dev_dataset)

if __name__=='__main__':
  try:
    from params import params
    from dataset import from_train_list
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    device = torch.device('cuda')
    print(device)
    print(f"Training configuration: {params.epochs} epochs, batch_size={params.batch_size}, lr={params.learning_rate}")

    train_list_path = "/data3/wangxiaopeng/code/Deepfake_Detection-main/data/label/train.txt"
    dev_list_path = "/data3/wangxiaopeng/code/Deepfake_Detection-main/data/label/dev.txt"  # Add dev dataset path
    audio_root = "/data3/wangxiaopeng/code/Deepfake_Detection-main/data/audio/train" # Add audio root path
    
    dataset = from_train_list(train_list_path, audio_root, params)
    dev_dataset = from_train_list(dev_list_path, audio_root, params)  # Create dev dataset
    
    model = Wav2Vec2_AASIST(layers=params.layers)
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    learner = Learner('./export', model, dataset, None, opt, params, dev_dataset=dev_dataset)

    # 自动恢复最新的checkpoint
    print("\n=== Checking for existing checkpoints ===")
    learner.list_all_checkpoints()
    learner.restore_from_checkpoint()

    # 测试单个batch
    print("\n=== Testing single batch ===")
    for features in dataset:
      wavs = features['waveforms']
      labels = features['labels']
      print(f"Waveforms shape: {wavs.shape}")
      print(f"Labels shape: {labels.shape}")
      print(f"Labels: {labels}")
      
      loss = learner.train_step(features, 10000)
      current_metrics = learner._get_current_metrics()
      learner._write_summary(10000, features, loss, current_metrics)
      
      print(f"Loss: {loss:.4f}")
      if current_metrics:
        print(f"Metrics: ACC={current_metrics['acc']:.4f}, AUC={current_metrics['auc']:.4f}, EER={current_metrics['eer']:.4f}")
        print(f"Real Loss: {current_metrics['real_loss']:.4f}, Fake Loss: {current_metrics['fake_loss']:.4f}")
      break
    
    # 开始完整训练
    print(f"\n=== Starting full training for {params.epochs} epochs ===")
    # learner.train()  # 取消注释以开始完整训练
      
  except KeyboardInterrupt:
    print("训练被用户中断")
  except Exception as e:
    print(f"训练过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
  finally:
    # 确保清理资源
    torch.cuda.empty_cache()
    print("资源清理完成")