## Labeling Knee Regions in X-ray Images Using LabelImg

### Install and Launch LabelImg
```
cd xgait
git clone https://github.com/HumanSignal/labelImg.git
cd labelImg
pip install -r requirements/requirements-linux-python3.txt
python labelImg.py
```

### Annotation Procedure
After launching LabelImg, a new window will appear.
Follow the steps below to annotate knee regions in X-ray images:
1. Clock `Open Dir`
'Open Dir'에서 라벨링할 이미지가 들어있는 폴더 선택: /mnt/d/xgait/yolo_finetune/labelImg_data/images/train
'Change Save Dir'에서 라벨링 결과를 저장할 폴더 선택: /mnt/d/xgait/yolo_finetune/labelImg_data/labels/train
'create recbox' 사용하여 bbox 만들기
'save' 누르고 'next image' 눌러서 폴더 안의 모든 이미지에 대해 라벨링 수행

/mnt/d/xgait/yolo_finetune/labelImg_data/images/test, /mnt/d/xgait/yolo_finetune/labelImg_data/labels/test에 대해서도 다시 수행

결과적으로 이미지와 라벨이 아래와 같은 구조로 저장됨.
```
ostdet
├── images
├──── train/003.jpg, ...
├──── test/001.jpg, ...
├── labels
├──── train/003.txt, ...
└──── test/001.txt, ...
```

## YOLO Fine-tuning 
yolo의 fine-tuning을 위해서는 `yolo_finetune_config.yaml` 안의 path, train, val 경로를 바꿔서 학습, test 이미지 변경 가능
python train.py \
    --epochs 50 \
    --output_pth /mnt/d/xgait/yolo_finetune/output \
    --exp_name knee_yolo11n

 처럼 옵션으로 학습 조건 변경

학습 파일은 --output_path 경로에 저장되며 따로 입력하지 않을 경우 default로 /mnt/d/xgait/yolo_finetune/runs/knee_yolo11n_ft에 저장됨

crop_knee.py 코드를 사용하여 특정 이미지 혹은 폴더 안의 이미들에 대해서 무릎 부분을 detect하고 무릎 부분만 정사각형으로 잘라내 저장할 수 있음

python crop_knee.py \
    --yolo_pth /mnt/d/xgait/yolo_finetune/output/knee_yolo11n/weights/best.pt \
    --input_pth /mnt/d/xgait/yolo_finetune/labelImg_data/images/test \
    --output_pth /mnt/d/xgait/yolo_finetune/croped_knee_imgs/test


python crop_knee.py \
    --yolo_pth /mnt/d/xgait/yolo_finetune/output/knee_yolo11n/weights/best.pt \
    --input_pth /mnt/d/xgait/yolo_finetune/labelImg_data/images/train \
    --output_pth /mnt/d/xgait/yolo_finetune/croped_knee_imgs/train

학습 결과에 대한 metric 검출
python test.py \
    --yolo_pth /mnt/d/xgait/yolo_finetune/output/knee_yolo11n/weights/best.pt \
    --config_pth /mnt/d/xgait/yolo_finetune/yolo_finetune_config.yaml \
    --output_path /mnt/d/xgait/yolo_finetune/output/eval \
    --exp_name knee_yolo11n
