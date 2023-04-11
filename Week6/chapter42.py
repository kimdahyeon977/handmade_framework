if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero import Variable
import dezero.functions as F

#토이 데이터셋
np.random.seed(0)
x= np.random.rand(100,1)
y= 5+2*x + np.random.rand(100,1)
x,y = Variable(x), Variable(y)

W= Variable(np.zeros((1,1)))
b= Variable(np.zeros(1))

def predict(x):
    y= F.matmul(x,W)+b
    return y

def mean_squared_error(x0,x1):
    diff = x0-x1
    return F.sum(diff**2)/len(diff)

lr = 0.1
iters=100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y,y_pred)
    
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    
    W.data -= lr * W.grad.data #인스턴스 변수의 data에 대해 계산해야한다.
    b.data -= lr * b.grad.data #단순히 데이터를 갱신할 뿐이므로 계산 그래프를 만들 필요는 없다.
    print(W,b,loss)

    