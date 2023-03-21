from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F

class Layer:
    def __init__(self):
        self._params = set() #Layer 인스턴스에 속한 매개변수 보관
    def __call__(self, *inputs): #입력받은 인수를 건네 forward 메서드를 호출한다.
        outputs = self.forward(*inputs)
        if not isinstance(outputs,tuple):
            outputs=(outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, inputs):
        raise NotImplementedError()
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            
            if isinstance(obj,Layer): #계속 재귀적으로 꺼낼 수 있다.
                yield from obj.params()
            else:
                yield obj
    def cleargrads(self): #모든 매개변수의 기울기를 재설정
        for param in self.params():
            param.cleargrad()
            
    def __setattr__(self, name, value): #인스턴스 변수를 설정할때 호출되는 특수 메서드 name인 인스턴스 변수에 값으로 value로 전달해준다.
        if isinstance(value,(Parameter,Layer)): #Layer안에 Layer들어가는 구조
            self._params.add(name) #Layer 인스턴스의 이름도 params에 추가된다.
        super().__setattr__(name,value)

class Linear(Layer):
    def __init__(self, out_size,  nobias= False, dtype = np.float32, in_size = None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name ='W')
        if self.in_size is not None: #in_size가 지정되어 있지 않다면 나중으로 연기
            self._init_W()
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype = dtype), name = 'b')
            
    def _init_W(self):
        I,O = self.in_size , self.out_size
        W_data = np.random.randn(I,O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = W_data
    def forward(self, x):
        #데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y= F.linear(x,self.W,self.b)
        return y