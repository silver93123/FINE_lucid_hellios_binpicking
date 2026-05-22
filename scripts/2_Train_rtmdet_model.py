"""Stage 3: RTMDet-Ins fine-tuning 실행 스크립트.

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/2_Train_rtmdet_model.py --dataset 20260521_114500

인자:
    --dataset   데이터셋 폴더명만 입력 (필수)
                공통 경로: /home/fine/FINE_RTMDet/data/dataset/
                예) 20260521_114500
                    → /home/fine/FINE_RTMDet/data/dataset/20260521_114500 로 자동 완성
    --config    MMDet config 파일 경로 (기본: configs/rtmdet-ins_bracket.py)
    --epochs    학습 epoch 수 (미지정 시 config 파일의 값 사용)

내부 동작:
    mmdet의 'tools/train.py' 로직을 그대로 호출.
    --dataset 폴더명으로 config의 data_root를 런타임에 override하므로
    config 파일을 직접 수정할 필요 없음.

학습 중 출력:
    - epoch별 loss
    - val_interval마다 mAP 평가
    - 체크포인트 저장 알림

학습 결과:
    work_dirs/rtmdet-ins_bracket_v1/
    ├── epoch_10.pth, epoch_20.pth, ...  ← 체크포인트
    ├── best_*.pth                        ← 최고 성능 모델
    ├── last_checkpoint                   ← 마지막 학습 위치
    ├── *.log                             ← 학습 로그
    └── vis_data/                         ← TensorBoard 로그
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]

# 데이터셋 공통 루트 경로 (고정)
DATASET_BASE = Path("/home/silver/binpicking_vision/FINE_RTMDet/data/dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTMDet-Ins fine-tuning 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
데이터셋 기본 경로: {DATASET_BASE}

예시:
  python scripts/2_Train_rtmdet_model.py --dataset 20260521_114500
  python scripts/2_Train_rtmdet_model.py --dataset 20260521_114500 --epochs 100
        """,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="FOLDER_NAME",
        help=f"데이터셋 폴더명만 입력 (공통 경로: {DATASET_BASE})\n"
             "예: 20260521_114500\n"
             "생략하면 사용 가능한 폴더 목록을 보여줍니다.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "rtmdet-ins_bracket.py",
        help="MMDet config 파일 경로 (기본: configs/rtmdet-ins_bracket.py)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="학습 epoch 수 (미지정 시 config 파일의 max_epochs 값 사용)",
    )
    return p.parse_args()


def prompt_dataset_selection() -> str:
    """DATASET_BASE 안의 폴더 목록을 보여주고 사용자가 선택하게 함."""
    if not DATASET_BASE.exists():
        print(f"ERROR: 데이터셋 기본 경로가 없습니다: {DATASET_BASE}", flush=True)
        sys.exit(1)

    folders = sorted([d.name for d in DATASET_BASE.iterdir() if d.is_dir()])
    if not folders:
        print(f"ERROR: {DATASET_BASE} 안에 폴더가 없습니다.", flush=True)
        sys.exit(1)

    print("=" * 70, flush=True)
    print(f"  데이터셋 폴더 목록: {DATASET_BASE}", flush=True)
    print("=" * 70, flush=True)
    for i, name in enumerate(folders, 1):
        print(f"  [{i:2d}] {name}", flush=True)
    print("-" * 70, flush=True)

    while True:
        try:
            raw = input("  번호 또는 폴더명 입력 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  취소됨.", flush=True)
            sys.exit(0)

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(folders):
                selected = folders[idx]
                print(f"  → {selected} 선택됨\n", flush=True)
                return selected
            else:
                print(f"  번호 범위를 벗어났습니다. 1~{len(folders)} 중 입력하세요.", flush=True)
        elif raw in folders:
            print(f"  → {raw} 선택됨\n", flush=True)
            return raw
        else:
            print(f"  '{raw}' 는 목록에 없습니다. 다시 입력하세요.", flush=True)


def resolve_dataset_dir(folder_name: str) -> Path:
    """폴더명을 DATASET_BASE와 결합하여 전체 경로 반환."""
    return DATASET_BASE / folder_name


def validate_dataset_dir(dataset_dir: Path) -> list[str]:
    """데이터셋 폴더 구조 검증. 문제 있으면 에러 메시지 리스트 반환."""
    errors = []
    if not dataset_dir.exists():
        errors.append(f"데이터셋 폴더가 없습니다: {dataset_dir}")
        return errors  # 폴더 자체가 없으면 하위 검사 불필요

    required = {
        "intensity/":                          "intensity 이미지 폴더",
        "annotations/instances_Train.json":    "COCO 형식 어노테이션 파일",
    }
    for rel, desc in required.items():
        if not (dataset_dir / rel).exists():
            errors.append(f"{desc} 없음: {dataset_dir / rel}")

    return errors


def main() -> int:
    args = parse_args()

    # ── --dataset 미입력 시 대화형 선택 ─────────────────────────────────────
    if args.dataset is None:
        args.dataset = prompt_dataset_selection()

    # ── 경로 확정 ────────────────────────────────────────────────────────────
    dataset_dir = resolve_dataset_dir(args.dataset)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config

    # ── 사전 검증 ────────────────────────────────────────────────────────────
    errors = validate_dataset_dir(dataset_dir)
    if errors:
        print("ERROR: 데이터셋 폴더 검증 실패", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        return 1

    if not config_path.exists():
        print(f"ERROR: config 파일이 없습니다: {config_path}", flush=True)
        return 1

    ann_file = str(dataset_dir / "annotations" / "instances_Train.json")

    # ── 헤더 출력 ────────────────────────────────────────────────────────────
    print("=" * 70, flush=True)
    print("  STEP 2: RTMDet-Ins fine-tuning", flush=True)
    print("=" * 70, flush=True)
    print(f"  Dataset:      {dataset_dir}", flush=True)
    print(f"  Config:       {config_path}", flush=True)
    print(f"  Project root: {ROOT}", flush=True)
    if args.epochs:
        print(f"  Epochs:       {args.epochs} (override)", flush=True)
    else:
        print(f"  Epochs:       config 파일 값 사용", flush=True)
    print("", flush=True)
    print("  실시간 진행 상황은 아래 로그 + work_dirs/.../*.log 파일에서 확인.", flush=True)
    print("  중단하려면 Ctrl+C", flush=True)
    print("=" * 70, flush=True)
    print("", flush=True)

    # ── mmdet 학습 실행 ──────────────────────────────────────────────────────
    import os
    os.chdir(ROOT)  # 상대경로 안전성을 위해 프로젝트 루트로 이동

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(str(config_path))

    # config 파일 수정 없이 데이터셋 경로를 런타임에 override
    data_root = str(dataset_dir) + "/"
    cfg.train_dataloader.dataset.data_root = data_root
    cfg.train_dataloader.dataset.ann_file  = ann_file
    cfg.val_dataloader.dataset.data_root   = data_root
    cfg.val_dataloader.dataset.ann_file    = ann_file
    cfg.val_evaluator.ann_file             = ann_file

    # epoch override (--epochs 지정 시)
    if args.epochs:
        cfg.train_cfg.max_epochs = args.epochs

    # work_dir 보장
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 이어서 학습 (incremental) ────────────────────────────────────────────
    # best_*.pth 가 있으면 → 기존 모델에서 이어서 학습
    # 없으면 (최초 1회)   → config의 load_from (coco 사전학습) 에서 시작
    _candidates = sorted(work_dir.glob("best_*.pth"))
    if _candidates:
        cfg.load_from = str(_candidates[-1])
        print(f"  이어서 학습:  {_candidates[-1].name}", flush=True)
    else:
        print(f"  최초 학습:    {Path(cfg.load_from).name}", flush=True)
    print("", flush=True)

    # Runner 생성 + 학습 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

    # 완료 후 최신 best 모델 확인
    _after = sorted(work_dir.glob("best_*.pth"))
    best_name = _after[-1].name if _after else "없음"

    print("\n" + "=" * 70, flush=True)
    print("  학습 완료", flush=True)
    print("=" * 70, flush=True)
    print(f"  체크포인트:   {work_dir}", flush=True)
    print(f"  Best 모델:    {best_name}", flush=True)
    print(f"\n  다음 학습 시 이 모델에서 자동으로 이어서 시작됩니다.", flush=True)
    print(f"  python scripts/2_Train_rtmdet_model.py --dataset <새_데이터셋>", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
