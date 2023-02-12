import numpy as np
class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs) #self.forward(x0, x1)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[Variable(Variable.as_array(y)) for y in ys] #반환 원소가 하나뿐이라면 해당 원소를 직접 반환
        
        for output in outputs:
            output.set_creator(self) #Add가 저장됨.
        self.inputs=inputs
        self.outputs= outputs
        # 리스트의 원소가 한개라면 첫번째 원소를 반환
        return outputs if len(outputs) >1 else outputs[0]
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
            gys = [output.grad for output in f.outputs] #outputs에 담긴 미분값들 리스트에 담기
            gxs = f.backward(*gys) #역전파 호출
            if not isinstance(gxs, tuple): #튜플로 변경
                gxs=(gxs, )
            for x,gx in zip(f.inputs, gxs): #역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장해둔다.
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)
    def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x
class Add(Function):
    def forward(self, x0, x1):
        y= x0+x1
        return y
    def backward(self, gy):
        return gy, gy
class Square(Function):
    def forward(self, x):
        y=x**2
        return y
    def backward(self, gy):
        x=self.inputs[0].data
        gx = 2*x*gy
        return gx
def add(x0, x1):
    return Add()(x0,x1)
def square(x):
    return Square()(x)
#코드 실행
#TODO : z= x^2 + y^2라는 계산을 z=add(suqare(x), square(y)로 풀어내기)
x=Variable(np.array(2.0))
y=Variable(np.array(3.0))

z=add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)