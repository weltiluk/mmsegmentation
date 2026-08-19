#!/usr/bin/env python3
from __future__ import annotations

"""Fine-tune a trained model on another dataset.

The checkpoint is loaded with ``load_from`` (weights only), not with
``--resume``.  Consequently, optimizer and learning-rate scheduler start
fresh for the fine-tuning phase.
"""

"""
python3 finetuning.py \
  --trained-model /workspaces/masterarbeit/mmsegmentation/results_seed42/osram_unet/real_dataset_splits_60_20_20 \
  --finetuning-dataset /workspaces/masterarbeit/masterarbeit/stratified_split/mixed_datasets/r5_s1_60_20_20_osram_two_encoders
  -- seed 42
"""


import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "osram_unet.py"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results_seed42" / "osram_unet"


def existing_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Verzeichnis nicht gefunden: {path}")
    return path


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Datei nicht gefunden: {path}")
    return path


def find_best_checkpoint(run_directory: Path) -> Path:
    candidates = sorted(run_directory.glob("best_fg_mDice_iter_*.pth"))
    if not candidates:
        candidates = sorted(run_directory.rglob("best_fg_mDice_iter_*.pth"))

    if not candidates:
        raise FileNotFoundError(
            "Kein best_fg_mDice_iter_*.pth unter " f"{run_directory} gefunden."
        )

    if len(candidates) > 1:
        names = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Mehrere beste Checkpoints gefunden. Bitte einen davon mit "
            f"--checkpoint angeben:\n  {names}"
        )

    return candidates[0].resolve()


def infer_seed(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"_seed(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-Tuning eines bereits trainierten Modells auf einem weiteren "
            "Datensatz."
        )
    )
    parser.add_argument(
        "--finetuning-dataset",
        required=True,
        type=existing_directory,
        help="Dataset-Root für das Fine-Tuning mit train/, validation/ und test/.",
    )

    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--trained-model",
        type=existing_directory,
        help=(
            "Verzeichnis des bereits trainierten Modells; "
            "best_fg_mDice_iter_*.pth wird automatisch "
            "gesucht."
        ),
    )
    checkpoint_group.add_argument(
        "--checkpoint",
        type=existing_file,
        help="Konkreter Checkpoint des bereits trainierten Modells (.pth).",
    )

    parser.add_argument(
        "--config",
        type=existing_file,
        default=DEFAULT_CONFIG,
        help=f"MMSeg-Basiskonfiguration (Standard: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Ausgabeordner; wird andernfalls automatisch benannt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed; wird andernfalls aus einem Namen mit _seedN abgeleitet.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5000,
        help="Fine-Tuning-Iterationen (Standard: 5000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Neu gestartete SGD-Lernrate (Standard: 0.001).",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=0.0001,
        help="Minimale PolyLR-Lernrate (Standard: 0.0001).",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=150,
        help="Validierungsintervall in Iterationen (Standard: 150).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Reguläres Checkpoint-Intervall (Standard: 1000).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Automatic Mixed Precision an tools/train.py weiterreichen.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Befehl nur anzeigen und nicht starten.",
    )
    parser.add_argument(
        "--allow-existing-work-dir",
        action="store_true",
        help="Training trotz bereits nicht-leerem Work-Dir starten.",
    )
    parser.add_argument(
        "--extra-cfg-options",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Zusätzliche MMSeg-Konfigurationsüberschreibungen.",
    )
    args = parser.parse_args()

    if args.max_iters <= 0:
        parser.error("--max-iters muss größer als 0 sein.")
    if args.lr <= 0 or args.eta_min < 0:
        parser.error("--lr muss > 0 und --eta-min muss >= 0 sein.")
    if args.eta_min > args.lr:
        parser.error("--eta-min darf nicht größer als --lr sein.")
    if args.val_interval <= 0 or args.checkpoint_interval <= 0:
        parser.error("Intervalle müssen größer als 0 sein.")

    return args


def main() -> int:
    args = parse_args()
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else find_best_checkpoint(args.trained_model)
    )

    seed = args.seed
    if seed is None:
        seed = infer_seed(args.trained_model or checkpoint)
    if seed is None:
        raise SystemExit(
            "Seed konnte nicht aus dem Pfad abgeleitet werden. "
            "Bitte --seed angeben."
        )

    for split in ("train", "validation", "test"):
        split_directory = args.finetuning_dataset / split
        if not split_directory.is_dir():
            raise SystemExit(f"Erwarteter Split fehlt: {split_directory}")

    if args.work_dir is None:
        source_name = (args.trained_model or checkpoint.parent).name
        dataset_name = args.finetuning_dataset.name
        work_dir = DEFAULT_RESULTS_DIR / "finetuned" / safe_name(
            # f"finetune_{source_name}_to_{dataset_name}_seed{seed}"
            f"finetune_{source_name}_to_{dataset_name}"
        )
    else:
        work_dir = args.work_dir.expanduser().resolve()

    if work_dir.is_dir() and any(work_dir.iterdir()):
        if not args.allow_existing_work_dir:
            raise SystemExit(
                f"Work-Dir ist bereits nicht leer: {work_dir}\n"
                "Anderen --work-dir wählen oder bewusst "
                "--allow-existing-work-dir setzen."
            )

    cfg_options = [
        f"data_root={args.finetuning_dataset}",
        f"load_from={checkpoint}",
        "resume=False",
        f"randomness.seed={seed}",
        f"train_cfg.max_iters={args.max_iters}",
        f"train_cfg.val_interval={args.val_interval}",
        f"optim_wrapper.optimizer.lr={args.lr}",
        f"param_scheduler.0.begin=0",
        f"param_scheduler.0.end={args.max_iters}",
        f"param_scheduler.0.eta_min={args.eta_min}",
        f"default_hooks.checkpoint.interval={args.checkpoint_interval}",
        *args.extra_cfg_options,
    ]

    command = [
        sys.executable,
        str(SCRIPT_DIR / "tools" / "train.py"),
        str(args.config),
        "--work-dir",
        str(work_dir),
    ]
    if args.amp:
        command.append("--amp")
    command.extend(["--cfg-options", *cfg_options])

    print("Fine-Tuning")
    print(f"  Fine-Tuning-Datensatz: {args.finetuning_dataset}")
    print(f"  Startgewichte: {checkpoint}")
    print(f"  Seed: {seed}")
    print(f"  Work-Dir: {work_dir}")
    print(f"  Iterationen: {args.max_iters}")
    print(f"  Lernrate: {args.lr} -> {args.eta_min}")
    print("\nBefehl:")
    print(shlex.join(command))

    if args.dry_run:
        return 0

    work_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=SCRIPT_DIR, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
