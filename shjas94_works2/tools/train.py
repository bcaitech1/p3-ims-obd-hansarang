import argparse
import copy
import os
import os.path as osp
import time
import warnings
from importlib import import_module
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.runner.optimizer import build_optimizer, OPTIMIZERS
from madgrad import MADGRAD
from adamp import AdamP
from mmcv.utils import get_git_hash
from mmcv.runner import load_checkpoint

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--config', type=str, default='/opt/ml/code/mmdetection_trash/configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py', help='train config file path')
    parser.add_argument('--work-dir', type=str, default='/opt/ml/code/mmdetection_trash/work_dirs/swin',
                        help='the dir to save logs and models')
    # parser.add_argument(
    #     '--resume-from', type=str, default='/opt/ml/code/mmdetection_trash/work_dirs/swin/latest.pth', help='the checkpoint file to resume from')
    parser.add_argument(
        '--resume-from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--checkpoint', type=str,
                        default=None, help='path of pretrained checkpoint')
    # parser.add_argument('--checkpoint', type=str,
    #                     default='/opt/ml/code/mmdetection_trash/work_dirs/swin/epoch_9.pth', help='path of pretrained checkpoint')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='set optimizer')
    parser.add_argument('--learning_rate', type=float,
                        default=5e-05, help='set learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-03, help='set weight decay for optimizer')

    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # logging argument
    parser.add_argument('--wandb_group', type=str, default='Default_group', help='group name for wandb project'
                        'type your model name')
    parser.add_argument(
        '--wandb_run_name', type=str, default='Default_run', help='run name for wandb project'
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # Set Optimizer
    # Set Wandb
    config_list = {
        'epoch': cfg.runner.max_epochs,
        'batch_size':  cfg.data.samples_per_gpu,
        'optimizer': cfg.optimizer,
        'optimizer_config': cfg.optimizer_config,
        'lr_config': cfg.lr_config
    }
    cfg.log_config.hooks[1].init_kwargs['project'] = 'shjas94'
    cfg.log_config.hooks[1].init_kwargs['entity'] = 'pstage3_det'
    cfg.log_config.hooks[1].init_kwargs['group'] = args.wandb_group
    cfg.log_config.hooks[1].init_kwargs['name'] = args.wandb_run_name
    cfg.log_config.hooks[1].init_kwargs['config'] = config_list
    ###############
    cfg.optimizer.lr = args.learning_rate
    cfg.optimizer.weight_decay = args.weight_decay
    cfg.runner.max_epochs = 20
    cfg.lr_config['policy'] = 'CosineAnnealing'
    cfg.lr_config['warmup'] = 'linear'
    cfg.lr_config['warmup_iters'] = 2
    cfg.lr_config['warmup_ratio'] = 1.0 / 10
    cfg.lr_config['min_lr_ratio'] = 1e-6

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu')

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
