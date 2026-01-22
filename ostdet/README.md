골관절염 분류 모델
2-class 혹은 3-class 지원.

OAI 데이터셋의 경우
2-class: positive / negative
3-class: positive / negative / implant
이때 2-class에서 implant 안의 pos / neg이 /mnt/d/xgait/dataset/OAI_test_implant.json, /mnt/d/xgait/dataset/OAI_train_implant.json에 분류되어 있음

GNU 데이터셋의 경우
2-class: 
3-class: 

train.py에는 config 파일 경로 넣어서 수행
test.py에서는 acc, precision, recall, f1-score 출력되도록 해줌

util.py에서는
logging 관리, 데이터셋 생성, batch 생성, metric 계산 등의 함수가 있음


python train.py --cfg_pth /mnt/d/xgait/ostdet/config/ostdet_2_class_config.yaml

python test.py --cfg_pth /mnt/d/xgait/ostdet/config/ostdet_2_class_config.yaml --ckpt_pth /mnt/d/xgait/ostdet/output/exp01/best.pth
