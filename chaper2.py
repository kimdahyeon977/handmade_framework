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
C=Square()
  
x=Variable(np.array(0.5))
y= A(B(A(x)))
print(y.data) 

#https://wjunsea.tistory.com/61