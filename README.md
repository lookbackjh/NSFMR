# 사용법
## 1. 파일 기능요약
`.py` 파일이 여러개인데 , 대충 data추출(`movielensdata.py`), negative sampling(`negativesampler.py`) , one hot encoding(`custompreprocess.py`)순서로 진행한다고 생각하면 됨. `recommender.py` 에서 훈련이랑 추천, testing동시에 되도록 구현해두었음, 이 데이터셋을 원래 만든사람이 5fold형태로 해놔서 , 5 fold cross validaiton이라고 생각하면 될듯.

`recommender.py` 그냥 실행시키면, 훈련, testing동시에 된다. 

## 2. 추가적으로 할일
현재 Uniform sampling이랑, 우리 방식의 Negative sampling 방식을 비교하고 있고, 앞으로 다른데이터셋에 적용한다던가, hyperparmeter( `alpha, beta, gamma, num_factor, topk `)등등 조절해보면서 할 필요가 있을듯. 다른 모델이랑도 비교할 필요가 있어보임 일단은 1차적으로 그뒤의 기능추가도 해야할 것 같고.
