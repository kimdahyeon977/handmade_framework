if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
import dezero.functions as F
from dezero import Variable

x= Variable(np.array([[1,2,3],[4,5,6]]))
c = Variable(np.array([[10,20,30],[40,50,60]]))
t= x+c
y = F.sum(t)
y.backward(retrain_grad = True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)