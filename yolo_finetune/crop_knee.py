import os
import glob
import argparse
import cv2
from ultralytics import YOLO


def make_square_box(x1, y1, x2, y2, img_w, img_h):
    """
    직사각 bbox(x1,y1,x2,y2)를 포함하는 가장 작은 정사각형 bbox 반환.
    이미지 경계를 넘지 않도록 좌표를 조정.
    """
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = min(side, img_w, img_h)

    sx1 = cx - side / 2.0
    sy1 = cy - side / 2.0
    sx2 = cx + side / 2.0
    sy2 = cy + side / 2.0

    if sx1 < 0:
        shift = -sx1
        sx1 += shift
        sx2 += shift
    if sx2 > img_w:
        shift = sx2 - img_w
        sx1 -= shift
        sx2 -= shift

    if sy1 < 0:
        shift = -sy1
        sy1 += shift
        sy2 += shift
    if sy2 > img_h:
        shift = sy2 - img_h
        sy1 -= shift
        sy2 -= shift

    sx1 = max(0, int(round(sx1)))
    sy1 = max(0, int(round(sy1)))
    sx2 = min(img_w, int(round(sx2)))
    sy2 = min(img_h, int(round(sy2)))

    return sx1, sy1, sx2, sy2


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def list_images(input_pth: str):
    """
    input_pth가
    - 파일이면: 그 파일 1개
    - 폴더면: 하위 포함 모든 이미지 파일
    """
    if os.path.isfile(input_pth):
        return [input_pth] if is_image_file(input_pth) else []
    # 폴더: 재귀적으로 이미지 수집
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.tif", "**/*.tiff", "**/*.webp"]
    imgs = []
    for p in patterns:
        imgs.extend(glob.glob(os.path.join(input_pth, p), recursive=True))
    imgs = sorted(set(imgs))
    return imgs


def process_image(model: YOLO, image_path: str, output_dir: str, conf_thres=0.3, imgsz=640,
                  save_debug=True, crop_size=224):
    """
    한 장의 이미지에 대해:
      - knee bbox 검출
      - 각 bbox → 정사각 bbox 확장
      - crop_size x crop_size crop 저장
      - (옵션) 원본 이미지/박스 표시 이미지 저장
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return 0

    h, w = img.shape[:2]

    results = model(img, imgsz=imgsz, conf=conf_thres, verbose=False)[0]

    if save_debug:
        cv2.imwrite(os.path.join(output_dir, f"{name}_orig.png"), img)
        img_draw = img.copy()
    else:
        img_draw = None

    crop_count = 0

    # 2개의 bbox가 검출된 경우, 왼쪽의 knee를 knee_L으로, 오른쪽의 knee를 knee_R으로
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        cx = (x1 + x2) / 2.0
        boxes.append((cx, x1, y1, x2, y2))

    if len(boxes) == 0:
        print(f"[INFO] No detections for: {image_path}")
        return 0

    boxes.sort(key=lambda b: b[0])

    if len(boxes) == 2:
        name_list = ["L", "R"]
    else:
        name_list = [f"knee{i}" for i in range(len(boxes))]

    for idx, ((_, x1, y1, x2, y2), knee_name) in enumerate(zip(boxes, name_list)):
        sx1, sy1, sx2, sy2 = make_square_box(x1, y1, x2, y2, w, h)

        crop = img[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(
            crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA
        )

        crop_save_path = os.path.join(output_dir, f"{name}_{knee_name}.png")
        cv2.imwrite(crop_save_path, crop_resized)
        crop_count += 1

        if save_debug and img_draw is not None:
            cv2.rectangle(img_draw, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)

    if save_debug and img_draw is not None:
        cv2.imwrite(os.path.join(output_dir, f"{name}_boxes.png"), img_draw)

    if crop_count == 0:
        print(f"[INFO] No detections for: {image_path}")
    else:
        print(f"[OK] {image_path} -> {output_dir} (crops={crop_count})")

    return crop_count


def parse_args():
    p = argparse.ArgumentParser(description="YOLO knee cropper (batch)")
    p.add_argument("--yolo_pth", type=str, required=True, help="Path to YOLO weights (.pt)")
    p.add_argument("--input_pth", type=str, required=True, help="Input image file or directory (recursive)")
    p.add_argument("--output_pth", type=str, required=True, help="Output directory to save crops")
    p.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    p.add_argument("--crop_size", type=int, default=224, help="Output crop size (square)")
    p.add_argument("--save_debug", action="store_true", help="Save *_orig.png and *_boxes.png")
    p.add_argument("--flat", action="store_true",
                   help="Do not mirror folder structure; save all outputs directly under output_pth")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.yolo_pth)

    images = list_images(args.input_pth)
    if not images:
        raise SystemExit(f"[ERROR] No images found under: {args.input_pth}")

    src_root = args.input_pth if os.path.isdir(args.input_pth) else os.path.dirname(args.input_pth)

    total_imgs = 0
    total_crops = 0

    for img_path in images:
        total_imgs += 1

        if args.flat:
            out_dir = args.output_pth
        else:
            rel_dir = os.path.relpath(os.path.dirname(img_path), src_root)
            out_dir = os.path.join(args.output_pth, rel_dir, os.path.splitext(os.path.basename(img_path))[0])

        total_crops += process_image(
            model=model,
            image_path=img_path,
            output_dir=out_dir,
            conf_thres=args.conf,
            imgsz=args.imgsz,
            save_debug=args.save_debug,
            crop_size=args.crop_size
        )

    print("====================================")
    print(f"Processed images: {total_imgs}")
    print(f"Total crops saved: {total_crops}")
    print(f"Output root: {args.output_pth}")


if __name__ == "__main__":
    main()
