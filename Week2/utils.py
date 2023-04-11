
##수동 역전파##
import numpy as np
class Variable:
    def __init__(self, data):
        #타입 에러 방지
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported!')
        self.data = data 
        self.grad = None #미분값이 들어갈것임
        self.creator = None #함수를 가져온다.
    def set_creator(self, func):
        self.creator = func
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f= funcs.pop()
            x,y = f.input, f.ouput
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)
    def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x
class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        self.input = input #self를 쓴 이유 : 입력 변수를 기억
        self.output = output
        return output
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self,x):
        y= x**2
        return y
    def backward(self, gy):
        x=self.input.data
        gx = 2*x*gy
        return gx

class Exp(Function):
    def forward(self, x):
        y=np.exp(x)
        return y
    def backward(self, gy):
        x=self.input.data
        gx= np.exp(x)*gy
        return gx
def square(x):
    return Square()(x)
def exp(x):
    return Exp()(x)

class Add(Function): #Function의 모든 인스턴스들을 상속받는다.
    def forward(self, xs):
        x0, x1= xs
        y=x0+x1
        return (y,)
'''
1. NotImplemented : 지원하지 않는 연산자이라고 알리기 위함. return None과 같음
모든 시도가 NotImplemented로 끝난다면, 그제서야 파이썬은 TypeError를 발생시킨다.

2. NotImplementedError : 호출되면 무조건 error 반환

3. super()__init__() : 부모 class의 모든 메소드를 불러오기 (주의! : 속성자체를 변화시키는 것은 아님!!)
출처 : https://supermemi.tistory.com/entry/Python-3-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%81%B4%EB%9E%98%EC%8A%A4%EC%9D%98-super-%EC%97%90-%EB%8C%80%ED%95%B4-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-superinit-super%EC%9D%98-%EC%9C%84%EC%B9%98
'''

# ##코드 실행##
# A=Square()
# B=Exp()
# C=Square()

# x=Variable(np.array(0.5))
# a=A(x)
# b=B(a)
# y=C(b)

# y.grad = np.array(0.1) #처음값은 dy/dy = 1로
# b.grad = C.backward(y.grad)
# a.grad = B.backward(b.grad)
# x.grad = A.backward(a.grad)
# print(x.grad)