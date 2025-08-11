import os
import json
import logging
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from learner import train_distributed_torchrun
import torch.distributed as dist
from params import params

# 在init_distributed_mode函数中确保使用正确的初始化方法
def init_distributed_mode(args):
    # 自动从环境变量获取多机参数
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    
    # 打印更多调试信息
    print(f"Initializing process group: rank={args.rank}, world_size={args.world_size}, gpu={args.gpu}")
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    
    # 确保所有进程同步
    dist.barrier()

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("No GPU device found for training.")

    init_distributed_mode(args=args)
    train_distributed_torchrun(args.gpu, args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('train_list', nargs='+',
      help='train list of phone, duration and mel ')
  parser.add_argument('audio_root', help='audio root')
#   parser.add_argument('train_list_unsupervised', nargs='+',
#       help='train list of mel')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  main(parser.parse_args())