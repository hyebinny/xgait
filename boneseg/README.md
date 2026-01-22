# Bone Segmentation

## Bone Segmentation with AttentionUNet
This module performs bone segmentation on lower-limb X-ray images using an Attention U-Net model.  
The task involves segmenting seven distinct bone structures from each input image.

### Train
To train the model, run the following command with the specified configuration file:
```bash
python boneseg/train.py --cfg_pth /mnt/d/xgait/boneseg/config/boneseg_config.yaml
```

### Test
During testing, the model is evaluated on the test set and Intersection over Union (IoU) scores are reported.  
If the `--visualize` option is enabled, the predicted segmentation masks are overlaid on the original images and saved for visualization.
```bash
python boneseg/test.py \
  --cfg_pth /mnt/d/xgait/boneseg/config/boneseg_config.yaml \
  --visualize
```
