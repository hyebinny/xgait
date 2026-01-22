import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO fine-tuning script")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        default="yolo_finetune/output",
        help="Directory to save training outputs (project path)"
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

    # Load pretrained model
    model = YOLO("xgait/yolo_finetune/yolo11n.pt")

    # Train
    results = model.train(
        data="yolo_finetune/yolo_finetune_config.yaml",
        epochs=args.epochs,
        imgsz=640,
        project=args.output_pth,
        name=args.exp_name,
        exist_ok=True
    )


if __name__ == "__main__":
    main()
