# Osteoarthritis Detection

## Osteoarthritis Classification
A pretrained backbone is loaded from `timm`, and the classification head is replaced to build and fine-tune an osteoarthritis classification model.  
Both 2-class and 3-class settings are supported.  
* 2-class: positive / negative
* 3-class: positive / negative / implant

In the 2-class setting, implant samples may be treated as either positive or negative depending on the annotation.  
The indices of implant samples labeled as negative are provided in:`dataset/[OAI/GNU]_[train/test]_implant.json`.

### Train
Training is performed according to the specified configuration file.  
Training can be conducted on either OAI or GNU, or on both datasets.
```bash
python ostdet/train.py --cfg_pth ostdet/config/ostdet_2_class_config.yaml
```

### Test
Evaluation is performed according to the specified configuration file.  
The following metrics are computed: Accuracy, Precision, Recall, and F1-score.  
Evaluation can be conducted on either OAI or GNU, or on both datasets.
```bash
python ostdet/test.py \
  --cfg_pth ostdet/config/ostdet_2_class_config.yaml \
  --ckpt_pth ostdet/output/exp01/best.pth
```
