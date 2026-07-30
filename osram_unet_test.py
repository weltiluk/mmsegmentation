"""Test the best OSRAM checkpoint without starting a training run.

python tools/test.py osram_unet_test.py

Only MODEL_DIR needs to be changed.
"""

from pathlib import Path as _Path

from mmengine.config import Config as _Config

_base_ = './osram_unet.py'

# Directory containing the trained model, its saved config and best checkpoint.
MODEL_DIR = (
    '/workspaces/masterarbeit/mmsegmentation/results/osram_unet/'
    'r1_s1_80_10_10_osram_more_rares_high_skip_no_transform_custom_dist')

_model_dir = _Path(MODEL_DIR)
if not _model_dir.is_dir():
    raise FileNotFoundError(f'Model directory does not exist: {_model_dir}')

_model_config_path = _model_dir / 'osram_unet.py'
if not _model_config_path.is_file():
    raise FileNotFoundError(
        f'No saved osram_unet.py found in: {_model_dir}')

# Read the test dataset location from the config saved during training.
_model_config = _Config.fromfile(_model_config_path)
try:
    data_root = str(_model_config.data_root)
except (AttributeError, KeyError) as _error:
    raise KeyError(
        f'No top-level data_root found in '
        f'{_model_config_path}') from _error

_best_checkpoints = sorted(_model_dir.glob('best_*.pth'))
if not _best_checkpoints:
    raise FileNotFoundError(
        f'No best_*.pth checkpoint found in: {_model_dir}')
if len(_best_checkpoints) > 1:
    _checkpoint_names = ', '.join(
        _path.name for _path in _best_checkpoints)
    raise RuntimeError(
        f'Multiple best checkpoints found in {_model_dir}: '
        f'{_checkpoint_names}. Please remove obsolete checkpoints.')

_model_name = _model_dir.name
load_from = str(_best_checkpoints[0])
work_dir = str(_model_dir.parents[2] / 'test_results' / _model_name)

# Runner.test() uses only this dataset. All remaining settings are inherited.
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='test/images',
            seg_map_path='test/masks',
        ),
    ),
)

# MMEngine serializes every remaining global as config. Remove helper objects
# that are needed only while this file is evaluated.
del _best_checkpoints
del _Config
del _model_config
del _model_config_path
del _model_dir
del _model_name
del _Path
