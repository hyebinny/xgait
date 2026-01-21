import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO IoU evaluation script")
    parser.add_argument(
        "--yolo_pth",
        type=str,
        required=True,
        help="Path to trained YOLO weights (best.pt)"
    )
    parser.add_argument(
        "--config_pth",
        type=str,
        required=True,
        default="/mnt/d/xgait/yolo_finetune/yolo_finetune_config.yaml",
        help="Path to dataset yaml file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Confidence threshold for evaluation"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold (used for mAP50 reporting)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/d/xgait/yolo_finetune/output/eval",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="knee_yolo11n",
        help="Experiment name"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load trained model
    model = YOLO(args.yolo_pth)

    # Run evaluation
    metrics = model.val(
        data=args.config_pth,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=args.output_path,
        name=args.exp_name,
        exist_ok=True
    )

    summary = (
        "====================================\n"
        "Evaluation finished\n"
        f"mAP50      : {metrics.box.map50:.4f}\n"
        f"mAP50-95   : {metrics.box.map:.4f}\n"
        f"Precision  : {metrics.box.mp:.4f}\n"
        f"Recall     : {metrics.box.mr:.4f}\n"
        "====================================\n"
    )

    # 콘솔 출력
    print(summary)

    # 결과 저장
    save_dir = os.path.join(args.output_path, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "result_summary.txt")
    with open(save_path, "w") as f:
        f.write(summary)

    print(f"[OK] Result summary saved to: {save_path}")


if __name__ == "__main__":
    main()
