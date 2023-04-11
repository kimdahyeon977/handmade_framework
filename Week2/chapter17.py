import weakref
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
        self.generation = 0 #세대 수를 기록하는 변수
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation +1 #세대를 기록한다(부모 세대 + 1)
    def backward(self):
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
def cleargrad(self): #미분값 초기화
    self.grad = None
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs) #self.forward(x0, x1)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys] #반환 원소가 하나뿐이라면 해당 원소를 직접 반환
        self.generation = max([x.generation for x in inputs]) #세대수가 높은 것을 선택
        for output in outputs:
            output.set_creator(self) #Add가 저장됨.
        self.inputs=inputs
        self.outputs= [weakref.ref(output) for output in outputs] #self.outputs가 대상을 약한 참조로 가리키게 변경 => 함수는 출력변수를 약하게 참조한다.
        # 리스트의 원소가 한개라면 첫번째 원소를 반환
        return outputs if len(outputs) >1 else outputs[0]
class Add(Function):
    def forward(self, x0, x1):
        y= x0+x1
        return y
    def backward(self, gy):
        return gy, gy
def add(x0, x1):
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
#순환 참소가 없어진 새로운 DeZero 코드 실행
for i in range(10):
    x=Variable(np.random.randn(10000)) #거대한 데이터
    y=square(square(square(x)))