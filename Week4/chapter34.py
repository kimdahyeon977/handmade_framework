if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

x= Variable(np.linspace(-7,7,200)) #-7부터 7까지 균일하게 200든분한 배열 만들어줌.
#다차원 배열을 입력 받으면 각 원소에 대해 독립적으로 계산. 따라서 한번의 계산으로 원소 200개의 계산이 모두 이루어진다.
y= F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx= x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    
#그래프 그리기
labels = ["y=sin(x)","y'", "y''" , "y'''"]
for i,v in enumerate(logs):
    plt.plot(x.data, logs[i] , label = labels[i])
plt.legend(loc='lower right')
plt.show()
