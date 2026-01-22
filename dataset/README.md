## Datasets
This project 3 datasets: OAI, GNU, and SNU.

OAI dataset download link: 
[https://www.kaggle.com/datasets/jeftaadriel/osteoarthritis-initiative-oai-dataset](https://www.kaggle.com/datasets/jeftaadriel/osteoarthritis-initiative-oai-dataset)

### OAI dataset

| Split | # Subjects | Positive (Ost) | Negative (Ost) |
|------:|-----------:|---------------:|---------------:|
| Train | 6,604      | 2,791          | 3,813          |
| Test  | 1,656      | 721            | 935            |

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

| Split | # Subjects | Positive (Ost) | Negative (Ost) | Normal (Align) | Abnormal (Align) |
|------:|-----------:|---------------:|---------------:|---------------:|-----------------:|
| Train | 34         | 0              | 34             | 18             | 16               |
| Test  | 14         | 1              | 13             | 7              | 7                |

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

| Split | # Subjects | Positive (Ost) | Negative (Ost) | Normal (Align) | Abnormal (Align) |
|------:|-----------:|---------------:|---------------:|---------------:|-----------------:|
| Train | 21         | 10             | 11             | 11             | 10               |
| Test  | 9          | 5              | 4              | 4              | 5                |

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
