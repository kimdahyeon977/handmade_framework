import numpy as np
import weakref
import contextlib
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs=[x.data for x in inputs]
        ys=self.forward(*xs) #self.forward(x0, x1)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys] #반환 원소가 하나뿐이라면 해당 원소를 직접 반환
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) #세대수가 높은 것을 선택
            for output in outputs:
                output.set_creator(self) #계산들의 '연결'을 만든다.
            self.inputs=inputs
            self.outputs= [weakref.ref(output) for output in outputs] #self.outputs가 대상을 약한 참조로 가리키게 변경 => 함수는 출력변수를 약하게 참조한다.
        # 리스트의 원소가 한개라면 첫번째 원소를 반환
        return outputs if len(outputs) >1 else outputs[0]
class Variable:
    __array_priority__ = 200 #우선순위 지정
    def __init__(self, data, name = None):
        #타입 에러 방지
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported!')
        self.data = data
        self.name = name  #변수의 이름을 붙여줄 수 있다.
        self.grad = None #미분값이 들어갈것임
        self.creator = None #함수를 가져온다.
        self.generation = 0 #세대 수를 기록하는 변수
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation +1 #세대를 기록한다(부모 세대 + 1)
    def backward(self,retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()
        def add_func(f): #DeZero 함수 리스트를 세대 순으로 정렬
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)
        while funcs:
            f= funcs.pop()
            gys = [output().grad for output in f.outputs] #outputs에 담긴 미분값들 리스트에 담기
            gxs = f.backward(*gys) #역전파 호출
            if not isinstance(gxs, tuple): #튜플로 변경
                gxs=(gxs, )
            for x,gx in zip(f.inputs, gxs): #역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장해둔다.
                if x.grad is None:
                    x.grad = gx #처음에는 그대로 대입
                else:
                    x.grad = x.grad + gx #그 다음번부터는 전달된 미분값을 더해줘야함.
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None #y는 약한 참조

    @property #메서드를 인스턴스 변수처럼 사용할 수있다.
    def shape2(self):
        return self.data.shape
    @property
    def ndim(self): #차원 수
        return self.data.ndim
    @property
    def size(self): #원소 수
        return self.data.size
    @property
    def dtype(self): #데이터 타입
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n', '\n'+' '*9) #좀더 보기편하게!
        return 'variable(' + p + ')'
    
def cleargrad(self): #미분값 초기화
    self.grad = None
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Add(Function):
    def forward(self, x0, x1):
        y= x0+x1
        return y
    def backward(self, gy):
        return gy, gy
def add(x0, x1):
    x1= as_array(x1)
    return Add()(x0,x1)
class Square(Function):
    def forward(self, x):
        y=x**2
        return y
    def backward(self, gy):
        x=self.inputs[0].data
        gx = 2*x*gy
        return gx
def square(x):
    return Square()(x) 

class Config:
    enable_backprop = True

def using_config(name, value):
    old_value = getattr(Config, name) #name을 getattr함수에 넘겨 Config클래스에서 꺼내온다.
    setattr(Config, name, value) #새로운 값을 설정
    try:
        yield
    finally:
        setattr(Config, name, old_value)
def no_grad():
    return using_config('enable_backprop',False)

class Mul(Function):
    def forward(self, x0, x1):
        y=x0*x1
        return y
    def backward(self, gy):
        x0,x1 = self.inputs[0].data, self.inputs[1].data
        return gy*x1 , gy*x0
class Neg(Function):
    def forward(self, x):
        return -x
    def backwrad(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0 , x1):
        y= x0-x1
        return y
    def backward(self, gy):
        return gy , -gy

class Div(Function):
    def forward(self, x0, x1):
        y= x0/x1
        return y
    def backward(self, gy):
        x0,x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0 , gx1
class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        y= x ** self.c
        return y
    def backward(self, gy):
        x= self.inputs[0].data
        c= self.c
        gx = c * x ** (c-1) * gy
        return gx
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1) #x0와 x1의 순서를 바꾼다.

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) #x0와 x1의 순서를 바꾼다.

def neg(x):
    return Neg()(x)

def mul(x0, x1):
    return Mul()(x0,x1)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0,x1)
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) #return할때는 x1, x0값을 바꾼다.
def pow(x, c):
    return Pow(c)(x)
#r~ 따로 지정하는 이유 : 나눗셈도 뺄셈과 마찬가지로 좌/우항 어느 것에 적용할지에 따라 적용되는 함수가 다르기때문
Variable.__add__ = add #Variable 인스턴스의 __add__메서드를 호출할때 add함수가 불린다.
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow

#코드 실행
x= Variable(np.array(2.0))
y = x**3
print(y)