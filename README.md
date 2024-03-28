# Youtube8M_torch


### 이미지

pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

### 추가 설치

protobuf==3.19.0

torchdata

#### import2.py 실행시켜도 됨.


### 참고사항
#### 완성도 낮음
#### eval.py : gap hit1 perr map 다 구현하긴 했지만 map이 높게 나와서 검증 필요
#### video와 frame py파일 분리 되어 있음
#### models에 모델 클래스를 정의해서 불러오는 방식


### 터미널 매개변수 X 
#### python frame_train.py 로 실행시켜야하니 실행마다 모델, 배치사이즈, 학습률, 에폭, 저장경로 등 직접 바꿔줘야함
