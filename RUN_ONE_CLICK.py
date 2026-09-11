from __future__ import annotations

"""ChemProcessRAG 一键入口。

默认：完整公开数据 benchmark（自动检查环境、自动下载、自动训练/测试、自动汇总）。

常用：
    python RUN_ONE_CLICK.py
    python RUN_ONE_CLICK.py --mode crystalcv
    python RUN_ONE_CLICK.py --mode selftest
    python RUN_ONE_CLICK.py --mode demo
    python RUN_ONE_CLICK.py --dry-run
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _banner(text: str) -> None:
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)


def _check_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(f"需要 Python 3.10+，当前为 {sys.version.split()[0]}")
    print(f"[OK] Python {sys.version.split()[0]}")


def _check_disk() -> None:
    total, used, free = shutil.disk_usage(ROOT)
    free_gb = free / (1024 ** 3)
    print(f"[INFO] 当前磁盘可用空间约 {free_gb:.1f} GB")
    if free_gb < 12:
        print("[WARN] 完整公开数据运行可能需要 >12 GB 临时/解压空间；建议准备 20 GB 以上。")


def _check_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        print("[INFO] PyTorch 尚未安装；完整 public runner 会尝试通过 requirements-public.txt 安装依赖。")
        return
    import torch
    print(f"[OK] torch {torch.__version__}")
    print(f"[INFO] CUDA available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU = {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] 未检测到 CUDA。程序仍可运行，但 HeinSight/TCPT 训练会明显更慢。")


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script)] + (extra or [])
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="ChemProcessRAG 一键运行入口")
    ap.add_argument(
        "--mode",
        choices=["full", "crystalcv", "selftest", "demo"],
        default="full",
        help="full=完整公开 benchmark；crystalcv=只跑 CrystalCV V2；selftest=工程自测；demo=演示视频管线",
    )
    ap.add_argument("--force", action="store_true", help="full 模式强制重跑已完成阶段")
    ap.add_argument("--epochs", type=int, default=None, help="crystalcv 模式 TCPT epoch 覆盖值")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="只做环境和入口检查，不下载/训练")
    args = ap.parse_args()

    _banner("ChemProcessRAG / TCPT — 一键入口")
    _check_python()
    _check_disk()
    _check_torch()

    required = [
        "RUN_PUBLIC_OVERNIGHT.py",
        "RUN_CRYSTALCV_V2.py",
        "RUN_SELF_TEST.py",
        "RUN_DEMO.py",
        "public_benchmark/downloads.py",
        "public_benchmark/crystalcv_prepare.py",
        "public_benchmark/crystalcv_benchmark.py",
        "chem_process_rag/pipeline.py",
        "tcpt/model.py",
    ]
    missing = [x for x in required if not (ROOT / x).exists()]
    if missing:
        raise FileNotFoundError("项目文件不完整：" + ", ".join(missing))
    print("[OK] 关键项目文件完整")

    if args.dry_run:
        print("[PASS] dry-run 完成：项目入口与基础环境检查通过。")
        return

    if args.mode == "selftest":
        _run("RUN_SELF_TEST.py")
    elif args.mode == "demo":
        _run("RUN_DEMO.py")
    elif args.mode == "crystalcv":
        extra = ["--seed", str(args.seed)]
        if args.epochs is not None:
            extra += ["--epochs", str(args.epochs)]
        _run("RUN_CRYSTALCV_V2.py", extra)
    else:
        extra = ["--seed", str(args.seed)]
        if args.force:
            extra.append("--force")
        _run("RUN_PUBLIC_OVERNIGHT.py", extra)


if __name__ == "__main__":
    main()
