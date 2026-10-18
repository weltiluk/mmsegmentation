"""Train/evaluate OSRAM DINOv3 with explicit datasets and separate outputs."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS = ROOT.parent / 'masterarbeit' / 'stratified_split'
CONFIG = ROOT / 'osram_dinov3_convnext.py'
WEIGHTS = ROOT / 'checkpoints' / 'dinov3-convnext-small-pretrain-lvd1689m'
DATASETS = {
    'baseline': SPLITS / 'real_dataset_splits_60_20_20',
    **{key: SPLITS / 'mixed_datasets' / f'r6_s1_60_20_20_osram_final20_{suffix}'
       for key, suffix in [('train60', 'train60'), ('right_kl', 'train60_right_kl'),
                           ('longer', 'train60_longer_training')]},
}


def train_config(dataset, seed=42, weights=WEIGHTS, amp=False):
    from mmengine.config import Config
    cfg = Config.fromfile(CONFIG)
    root = DATASETS[dataset]
    cfg.data_root = str(root)
    cfg.dataset_name = root.name
    for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        cfg[loader].dataset.data_root = str(root)
    cfg.model.backbone.pretrained = str(Path(weights).resolve())
    cfg.randomness.seed = seed
    cfg.work_dir = str(ROOT / 'results_dinov3' / f'seed{seed}' / root.name)
    cfg.launcher = 'none'
    if amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
    return cfg


def verify_data(cfg, training):
    counts = {}
    loaders = ('train_dataloader', 'val_dataloader') if training else ('test_dataloader',)
    for key in loaders:
        dataset = cfg[key].dataset
        root = Path(dataset.data_root)
        images = {p.stem for p in (root / dataset.data_prefix.img_path).glob('*.png')}
        masks = {p.stem for p in (root / dataset.data_prefix.seg_map_path).glob('*.png')}
        if not images or images != masks:
            raise RuntimeError(f'Missing or mismatched images/masks in {root} ({key})')
        counts[key] = len(images)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    train = subs.add_parser('train')
    train.add_argument('--dataset', choices=DATASETS, default='baseline')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--weights', type=Path, default=WEIGHTS)
    train.add_argument('--amp', action='store_true')
    train.add_argument('--resume', action='store_true')
    train.add_argument('--dry-run', action='store_true')
    test = subs.add_parser('test')
    test.add_argument('model_dir', type=Path)
    test.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    from mmengine.config import Config
    if args.command == 'train':
        cfg = train_config(args.dataset, args.seed, args.weights, args.amp)
        directory = Path(cfg.work_dir)
        saved = directory / CONFIG.name
        if args.resume:
            if not saved.is_file() or not (directory / 'last_checkpoint').is_file():
                parser.error('Resume requires a saved config and last_checkpoint.')
            cfg = Config.fromfile(saved)
            cfg.resume = True
            cfg.model.backbone.pretrained = None  # full checkpoint restores encoder
        elif directory.exists() and any(directory.iterdir()):
            parser.error(f'Output already exists: {directory}. Use --resume for that run.')
        counts = verify_data(cfg, training=True)
        if not args.dry_run and not args.resume:
            source = Path(cfg.model.backbone.pretrained)
            if not (source / 'config.json').is_file() or not list(source.glob('*.safetensors')):
                parser.error('Pretrained weights are missing. Run tools/prepare_osram_dinov3.py first.')
    else:
        directory = args.model_dir.resolve()
        saved = directory / CONFIG.name
        cfg = Config.fromfile(saved)
        checkpoints = sorted(directory.glob('best_fg_mDice_iter_*.pth'))
        if len(checkpoints) != 1:
            parser.error(f'Expected exactly one best fg_mDice checkpoint in {directory}.')
        cfg.load_from = str(checkpoints[0])
        cfg.resume = False
        # The full MMSeg checkpoint contains the backbone, enabling offline testing.
        cfg.model.backbone.pretrained = None
        cfg.work_dir = str(ROOT / 'test_results_dinov3' / f'seed{cfg.randomness.seed}' / directory.name)
        cfg.test_evaluator = [
            dict(type='ClasswiseIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
            dict(type='HausdorffDistanceMetric', percentile=95, include_background=False)]
        counts = verify_data(cfg, training=False)
    print(json.dumps(dict(mode=args.command, data_root=cfg.data_root,
                         work_dir=cfg.work_dir, seed=cfg.randomness.seed,
                         counts=counts, pretrained=cfg.model.backbone.pretrained,
                         frozen=cfg.model.backbone.frozen), indent=2))
    if args.dry_run:
        return
    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    if args.command == 'train':
        runner.train()
    else:
        runner.test()


if __name__ == '__main__':
    main()
