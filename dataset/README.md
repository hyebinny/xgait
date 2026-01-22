## Datasets
This project 3 datasets: OAI, GNU, and SNU.

OAI dataset download link: 
[https://www.kaggle.com/datasets/jeftaadriel/osteoarthritis-initiative-oai-dataset](https://www.kaggle.com/datasets/jeftaadriel/osteoarthritis-initiative-oai-dataset)

### OAI dataset
```
dataset/OAI
├── train
├──── positive
├──── negative
├──── implant
├── test
├──── positive
├──── negative
├──── implant
├── OAI_json
├──── OAI_train_implant.json    # Indices of implant samples labeled as negative
└──── OAI_test_implant.json
```

### GNU dataset
```
dataset/GNU
├── 001                  # Left and right knee regions cropped from X-ray images using the fine-tuned YOLO model
├──── knee   
├────── 001_L.png
├────── 001_R.png
├──── xray/001.jpg       # Full lower limb xray image
├──── label/001.json     # Bone segment labels of lower limb xray image
├──── dcm/001.dcm        # Raw .dcm files
├── 002
...
├── GNU_json
├──── GNU_align.json         # align: normal / anormal
├──── GNU_ost.json           # ost: postiive / negative / implant
├──── GNU_split.json         # train / test split
├──── GNU_train_implant.json
└──── GNU_test_implant.json
```

### SNU dataset
```
dataset/SNU
├── 001
├──── knee/001.png
├──── xray/001.jpg 
├──── label/001.json 
├──── dcm/001.dcm
├── 002
...
├── SNU_json
├──── GNU_align.json
├──── GNU_ost.json
├──── GNU_split.json
├──── GNU_train_implant.json
├──── GNU_test_implant.json
└──── GNU_test_implant.json  # Gait information
```
