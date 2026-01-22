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
1. Click `Open Dir` and select the directory containing training images: `/mnt/d/xgait/yolo_finetune/labelImg_data/images/train`
2. Click  `Change Save Dir` and select the directory to save annotation files: `/mnt/d/xgait/yolo_finetune/labelImg_data/labels/train`
3. Use `Create RectBox` to draw bounding boxes around knee regions.
4. Click `Save`, then `Next Image` to repeate the process for all images in the `Open Dir` folder.
5. Repeat the same procedure for the test set:
```
Open Dir: /mnt/d/xgait/yolo_finetune/labelImg_data/images/test
Save Dir: /mnt/d/xgait/yolo_finetune/labelImg_data/labels/test
```

### Resulting Directory Structure
```
ostdet/labelImg_data
├── images
├──── train/003.jpg, ...
├──── test/001.jpg, ...
├── labels
├──── train/003.txt, ...
└──── test/001.txt, ...
```

## YOLO Fine-tuning 
### Train
`ultralytics` 에서 제공하는 YOLO fine-tuning 코드를 활용하여 학습 및 test를 수행함.
아래 command를 사용하여 yolo의 fine-tuning을 진행할 수 있고, `yolo_finetune_config.yaml` 안의 내용을 수정하여 fine-tuning condition을 설정할 수 있음.
`path`: labelImg를 사용하여 라벨링한 이미지-annotation 쌍의 root 경로
`train`: train에 사용하는 이미지들의 경로
`val`: test에 사용하는 이미지들의 경로
`test`: test에 사용하는 이미지들의 경로

```bash
cd xgait/yolo_finetune
python train.py \
    --epochs 50 \
    --output_pth output \
    --exp_name knee_yolo11n
```

학습 결과는 --output_path 경로에 저장되며 따로 입력하지 않을 경우 default로 /yolo_finetune/output/[exp_name] 안에 저장됨.

### Test
학습 결과에 대한 metric 검출
python test.py \
    --yolo_pth /mnt/d/xgait/yolo_finetune/output/knee_yolo11n/weights/best.pt \
    --config_pth /mnt/d/xgait/yolo_finetune/yolo_finetune_config.yaml \
    --output_path /mnt/d/xgait/yolo_finetune/output/eval \
    --exp_name knee_yolo11n

### Crop Images
crop_knee.py 코드를 사용하여 특정 이미지 혹은 폴더 안의 이미들에 대해서 무릎 부분을 detect하고 무릎 부분만 정사각형으로 잘라내 저장할 수 있음
python crop_knee.py \
    --yolo_pth [fine_tuned_yolo_pth].pt \
    --input_pth labelImg_data/images/test \
    --output_pth croped_knee_imgs/test
