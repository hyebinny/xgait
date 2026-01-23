# Skeletal Anomaly Detection

## Skeletal Anomaly Detection
A two-class classification model that distinguishes between normal and abnormal conditions in lower-limb X-ray images.
A pretrained backbone is loaded from `timm`, and the classification head is replaced to build and fine-tune an anoamly classification model.  

### Train
Training is performed according to the specified configuration file.  
Training can be conducted on either OAI or GNU, or on both datasets.
```bash
python anomalydet/train.py --cfg_pth anomalydet/config/anomalydet_config.yaml
```

### Test
Evaluation is performed according to the specified configuration file.  
The following metrics are computed: Accuracy, Precision, Recall, and F1-score.  
Evaluation can be conducted on either OAI or GNU, or on both datasets.
```bash
python anomalydet/test.py \
  --cfg_pth anomalydet/config/anomalydet_config.yaml \
  --ckpt_pth [ckpt_pth].pth
```
