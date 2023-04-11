if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero import Variable

def f(x):
    y= x ** 4 - 2 * x **2
    return y
def gx2(x): #2차 미분을 계산하기 위해 수식을 손으로 썼음.
    return 12 * x ** 2 -4

x= Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i,x)
    
    y= f(x)
    x.cleargrad()
    y.backward()
    
    x.data -= x.grad / gx2(x.data) 