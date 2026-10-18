"""Download the official gated HF snapshot without starting training."""
import argparse
import json
from pathlib import Path

MODEL_ID = 'facebook/dinov3-convnext-small-pretrain-lvd1689m'
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints' / MODEL_ID.split('/')[-1])
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    try:
        revision = HfApi().model_info(MODEL_ID).sha
        snapshot_download(
            MODEL_ID, revision=revision, local_dir=args.output,
            allow_patterns=['config.json', 'preprocessor_config.json', '*.safetensors',
                            '*.safetensors.index.json'])
    except GatedRepoError:
        parser.exit(2, f'Access to https://huggingface.co/{MODEL_ID} is required.\n'
                    'Accept/request access there, then run .venv/bin/hf auth login '
                    'locally and repeat this command. Do not paste tokens into chat.\n')
    except HfHubHTTPError as exc:
        parser.exit(2, f'Hugging Face download failed ({type(exc).__name__}). '
                    'Check connectivity and model access.\n')
    config = json.loads((args.output / 'config.json').read_text())
    if config.get('depths') != [3, 3, 27, 3] or config.get('hidden_sizes') != [96, 192, 384, 768]:
        raise RuntimeError('Downloaded architecture is not DINOv3 ConvNeXt-Small.')
    if not list(args.output.glob('*.safetensors')):
        raise RuntimeError('No safetensors weights were downloaded.')
    (args.output / 'download_manifest.json').write_text(json.dumps(
        {'model_id': MODEL_ID, 'revision': revision}, indent=2))
    print(f'Pretrained snapshot ready: {args.output}')


if __name__ == '__main__':
    main()
