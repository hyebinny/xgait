# Bone Segmentation

## Bone Segmentation with AttentionUNet
AttentionUNet 모델을 사용하여 lower limb xray image로부터 7개의 bone을 segmentation.

### Train
```bash
python boneseg/train.py --cfg_pth /mnt/d/xgait/boneseg/config/boneseg_config.yaml
```

### Test
Test set에 대해서 IoU 계산하여 report.
--visualize 인자를 주면 원본 이미지에 segmentation mask가 그려진 그림 저장됨.
```bash
python boneseg/test.py \
  --cfg_pth /mnt/d/xgait/boneseg/config/boneseg_config.yaml \
  --visualize
```
