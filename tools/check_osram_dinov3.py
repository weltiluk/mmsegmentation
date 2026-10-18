"""CPU integration check: real pipeline, decoder update, frozen backbone, padding.

--random-weights checks wiring only; it is not a pretrained-model evaluation.
No training checkpoint or segmentation scores are produced.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-weights', action='store_true')
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    import torch
    from mmengine.dataset import pseudo_collate
    from mmengine.logging import MMLogger
    from mmengine.structures import PixelData
    from mmseg.registry import DATASETS, MODELS
    from mmseg.structures import SegDataSample
    from mmseg.utils import register_all_modules
    from tools.osram_dinov3 import WEIGHTS, train_config
    MMLogger.get_instance('dinov3_check', log_level='WARNING')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    register_all_modules()
    cfg = train_config('baseline', weights=args.weights or WEIGHTS)
    if args.random_weights:
        cfg.model.backbone.pretrained = None
    model = MODELS.build(cfg.model).cpu()
    model.init_weights()
    model.train()
    assert not model.backbone.encoder.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    batch = model.data_preprocessor(pseudo_collate([dataset[0]]), training=True)
    assert batch['inputs'].shape == (1, 3, 512, 512)
    with torch.no_grad():
        feats = model.backbone(batch['inputs'])
        reference = model.backbone.encoder(batch['inputs'], output_hidden_states=True)
    assert [tuple(f.shape) for f in feats] == [
        (1, 96, 128, 128), (1, 192, 64, 64),
        (1, 384, 32, 32), (1, 768, 16, 16)]
    for feature, expected in zip(feats[:3], reference.hidden_states[1:4]):
        torch.testing.assert_close(feature, expected)
    # HF token zero is pooled; the remaining normalized tokens are dense features.
    expected = reference.last_hidden_state[:, 1:].transpose(1, 2).reshape_as(feats[-1])
    torch.testing.assert_close(feats[-1], expected)
    before = next(model.backbone.parameters()).detach().clone()
    decoder_before = model.decode_head.conv_seg.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    losses = model.loss(**batch)
    loss = sum(v for k, v in losses.items() if 'loss' in k)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.decode_head.conv_seg.weight.grad.abs().sum() > 0
    optimizer.step()
    torch.testing.assert_close(next(model.backbone.parameters()), before, rtol=0, atol=0)
    assert not torch.equal(model.decode_head.conv_seg.weight, decoder_before)
    # Non-multiple-of-32 image: check padding and restoration to native dimensions.
    sample = SegDataSample(metainfo=dict(
        ori_shape=(501, 503), img_shape=(501, 503), pad_shape=(501, 503),
        scale_factor=(1., 1.), flip=False))
    sample.gt_sem_seg = PixelData(data=torch.zeros(1, 501, 503, dtype=torch.long))
    inputs = torch.randint(0, 256, (3, 501, 503), dtype=torch.uint8)
    model.eval()
    with torch.no_grad():
        test_batch = model.data_preprocessor(dict(inputs=[inputs], data_samples=[sample]), training=False)
        pred = model.predict(**test_batch)
    assert tuple(pred[0].pred_sem_seg.data.shape) == (1, 501, 503)
    assert set(pred[0].pred_sem_seg.data.unique().tolist()) <= set(range(5))
    print(json.dumps(dict(
        status='passed', pretrained=not args.random_weights,
        pipeline_input=list(batch['inputs'].shape),
        feature_shapes=[list(f.shape) for f in feats],
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        finite_loss=float(loss.detach()), backbone_unchanged=True,
        decoder_updated=True, native_prediction_shape=list(pred[0].pred_sem_seg.data.shape)), indent=2))


if __name__ == '__main__':
    main()
