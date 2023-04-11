if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가


import numpy as np
from dezero import Variable

import math

def my_sin(x, threshold = 0.0001): #threshold는 정밀도
		y=0
		for i in range(100000):
				c = (-1)** i / math.factorial(2 * i +1)
				t = c * x ** (2 * i + 1)
				y = y + t
				if abs(t.data) < threshold:
						break
		return y 

#코드 실행
x=Variable(np.array(np.pi/4))
y=my_sin(x)
y.backward()

print(y.data) #0.707106
print(x.grad) #0.7071032