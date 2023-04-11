'''
< __init__과 __call__의 차이점 >
1. __init__
인스턴스를 초기화 할때 사용된다.

( 사용 예시 )
def __init__(self, n1, n2):
    self.n1= n1
    self.n2= n2
return 값을 사용하지 않음에 주의

2. __call__
클래스의 객체를 호출하게 만들어주는 메서드
즉, 인스턴스가 호출되었을떄 실행된다.

( 사용 예시 )
def __call__(self, n1, n2):
    self.n1 = n1
    self.n2 = n2
    return print(self.n1 + self.n2 )

s=Calc(1,2)

'''
from chaper1 import Variable
import numpy as np
class Function:
    def __call__(self, input):
        x= input.data #데이터를 꺼낸다.
        y= self.forward(x)
        output=Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()

class Square(Function): #Function의 클래스를 상속하는 클래스
    def forward(self, x):
        return x**2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)


A=Square()
B=Exp()

  
x=Variable(np.array(0.5))
y= A(B(A(x)))
print(y.data) 

#출처 : https://wjunsea.tistory.com/61

