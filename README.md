# 사용법

## 1. Negative Sample Generator.
negative sample이 어떻게 만들어지는지, 파일형태로 만들어서 train을 돌리는데 형딴에서 training돌려보고 싶으면, 
`datagenrator.ipynb` 전부 다돌리면 `dataset/ml-100k/data_one_hot.pkl` 형태로 파일을 만들수 있음
원래 칼럼이  user_id|movie_id|rating|movie_frequency|user_frequency  로 구성이 되는데, (여기서 rating은 implicit feedback이니까 그냥 1or 0으로 생각)
모든 user_id, 모든 movie_id에 대해서 one hot encoding시켜주고, 여기서 추가로 각각의 movie의 특성(장르) user의 특성(나이, 성별, 직업) 도 추가해준 column들이 들어가게됨(`datagenrator.ipynb` 돌리면서 확인해보면 좋을듯)

## 2. Trainer
`recommender.py`위에서 만들어진 pkl파일 통해서 바로 훈련 돌릴 수 있게 해놨음 epoch 500줬는데 300만 줘도 될듯 parse.arg.maxepoch만 바꿔줍면 됨. 

## 3. Test 

조금 막히는 부분이 여기인데, 
이게 test가 어떤 식으로 진행되냐면, 
모든 유저별로 모든 movie에 대해서 그 movie에 대한 선호도(점수)를 산출해서 상위 5개를 산출함, 그리고 그 상위 5개가 test에 있는지 확인하는 로직인데
일단 test할 수 있게끔 데이터 만드는 과정은 `testgenrator.ipynb` 에 들어가 있음. 
이런식으로 구현해서 평가한 논문을 찾기가 쉽지않아서 일단 각유저별로 추천상위5개 를 추천하도록 구현은 해두었음
여기서 이미 봤던영화를 포함시켜야할지 말지등등을 좀더 고려해봐야할것같음. 

## 4. 첨언

추가로 막히거나 해야할일 정리는 issue라인에 남겨둘테니 형 함 확인하면서 모르는부분이나 같이 확인해볼만한 부분은 issue에 남겨주면 될듯.
내일 밤이나 일요일 오후는 프리해서 그때 까지 또 진행해보고 있을게.  
