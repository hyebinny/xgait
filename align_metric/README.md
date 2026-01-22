# Extract Alignment Metrics

## Alignment Metric
The model trained in `/boneseg` is loaded to perform segmentation on X-ray .jpg images, after which alignment metrics are extracted.  
The path to the pretrained bone segmentation model must be specified in the configuration file.  

If the `--visualize` option is enabled, images are saved with the segmentation results and key joint points overlaid on the original X-ray image.  

```bash
python align_metric/extract_align_metrics.py \
  --cfg_pth align_metric/config/align_metric_config.yaml \
  --visualize
```
