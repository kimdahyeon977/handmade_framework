if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 #발걸음 보폭
iters = 50000 #반복 횟수

for i in range(iters):
    print(x0, x1)
    y= rosenbrock(x0, x1)
    x0.cleargrad() #x0.grad와 x1.grad에 미분값이 계속 누적되기 때문에 새롭게 미분할때는 지금까지 누적된 값을 초기화해야한다.
    x1.cleargrad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
y= rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)
#기울기는 각 지점에서 함수의 출력을 가장 크게 하는 방향을 가리킴.
#반대로 기울기에 마이너스를 곱한 (2, -400) 방향은 y값을 가장 작게 줄여주는 방향.

    
