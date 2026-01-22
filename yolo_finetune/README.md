## Labeling Knee Regions in X-ray Images Using LabelImg

### Install and Launch LabelImg
```
cd xgait
git clone https://github.com/HumanSignal/labelImg.git
python labelImg/labelImg.py
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
YOLO fine-tuning was performed using the training pipeline provided by ultralytics.
Both training and evaluation were conducted based on this implementation.
Fine-tuning can be executed using the command below, and the training conditions can be configured by modifying the contents of `yolo_finetune_config.yaml`.

```bash
cd xgait
python yolo_finetune/train.py \
    --epochs 50 \
    --output_pth output \
    --exp_name knee_yolo11n
```

The training results are saved to the directory specified by --output_pth.
If not explicitly provided, results are stored by default in `yolo_finetune/output/[exp_name]`.

### Test
To evaluate the trained model and compute detection metrics, run:
```bash
python yolo_finetune/test.py \
    --yolo_pth [fine_tuned_yolo_pth].pt \
    --config_pth yolo_finetune/yolo_finetune_config.yaml \
    --output_path yolo_finetune/output/eval \
    --exp_name knee_yolo11n
```

### Crop Images
The `crop_knee.py` script detects knee regions in a single image or in all images within a directory and crops the detected knee regions into square patches.
```bash
python yolo_finetune/crop_knee.py \
    --yolo_pth [fine_tuned_yolo_pth].pt \
    --input_pth yolo_finetune/labelImg_data/images/test \
    --output_pth yolo_finetune/croped_knee_imgs/test
```
