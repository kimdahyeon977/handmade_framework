if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.models import MLP

def softmax1d(x):
    x= as_variable(x) #ndarray일 경우 Variable 인스턴스로 변환
    y= F.exp(x)
    sum_y = F.sum(y)
    return y/sum_y
model = MLP((10,3)) #2층으로 이루어진 완전 연결 신경망을 만들어준다.
#첫번째 완전 연결 계층의 출력 크기는 10, 두번째 완전 연결 계층의 출력크기는 3.

x= np.array([[0.2,-0.4], [0.3,0.5], [1.3,-3.2], [2.1,0.3]])
t= np.array([2,0,1,0])
y=model(x)
loss= F.softmax_cross_entropy_simple(y,t)
print(loss)