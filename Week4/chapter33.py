if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':

    x = Variable(np.array(1.0))

		# 이 시점에서 순전파 및 역전파 계산 그래프 생성
    y = F.sin(x)
    y.backward(create_graph=True)

		# 2, 3, 4차 미분값 출력
    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad)