if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import dezero.layers as L
import dezero.functions as F
from dezero.layers import Layer
import numpy as np
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

np.random.seed(0)
x= np.random.rand(100,1)
y= np.sin(2*np.pi*x) + np.random.rand(100,1)
lr= 0.2
max_iter= 10000
hidden_size=10

model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model) #model의 모든 params를 넘기기


#학습 시작
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y,y_pred)
    
    model.cleargrads() #매개변수의 기울기 재설정
    loss.backward()
    
    optimizer.update() #매개변수 갱신 완료
    if i%1000==0:
        print(loss)