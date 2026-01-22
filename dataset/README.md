## Datasets
This project 3 datasets: OAI, GNU, and SNU.

OAI dataset download link: 
[https://www.kaggle.com/datasets/jeftaadriel/osteoarthritis-initiative-oai-dataset](https://www.kaggle.com/datasets/jeftaadriel/osteoarthritis-initiative-oai-dataset)

### OAI dataset
```
dataset/OAI
├── train            # knee images
├── test
├── OAI_split.json
├── OAI_implant_negative.json
```

### GNU dataset
```
dataset/GNU
├── 001
├──── knee/001.png    # fine-tuned YOLO를 사용하여 xray 이미지로부터 잘라낸 왼쪽과 오른쪽 무릎
├──── xray/001.jpg    # Full lower limb xray image
├──── label/001.json  # Lower limb image에 대한 bone segment label 정보
├──── dcm/001.dcm     # xray 이미지 원본 dcm 파일
├── 002
...
├── GNU_split.json
├── GNU_ost.json
```

### SNU dataset
```
dataset/SNU
├── 001
├──── knee/001.png    # fine-tuned YOLO를 사용하여 xray 이미지로부터 잘라낸 왼쪽과 오른쪽 무릎
├──── xray/001.jpg    # Full lower limb xray image
├──── label/001.json  # Lower limb image에 대한 bone segment label 정보
├──── dcm/001.dcm     # xray 이미지 원본 dcm 파일
├── 002
...
├── SNU_split.json
├── SNU_ost.json
├── SNU_gait.json     # subject의 gait label
```