if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph = True) #미분을 하기 위해 역전파하는 코드
gx = x.grad #단순한 변수가 아니라 계산그래프 (시작) , 따라서 x.grad의 계산 그래프에 대해 추가로 역전파 가능

x.cleargrad()

z = gx ** 3 + y
z.backward()
print(f'gx: {gx}')




