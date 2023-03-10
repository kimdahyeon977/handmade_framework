if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero import Variable
import dezero.functions as F

x=Variable(np.array([[1,2,3],[4,5,6]]))
y=F.reshape(x,(6,))
y.backward(retain_grad = True) #x의 기울기를 구한다.
print(x.grad)

x= Variable(np.array([[1,2,3],[4,5,6]]))
y= F.transpose(x)
y.backward()
print(x.grad)