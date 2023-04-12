from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F
import os
from dezero import cuda
                    
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
    def _flatten_params(self, params_dict, parent_key= ''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict , key)
            else:
                params_dict[key] = obj
    def save_weights(self, path):
        self.to_cpu()
        params_dict ={}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    def load_weights(self, path):
        npz = np.load(path)
        params_dict={}
        self._flatten_params(params_dict)
        for key,param in params_dict.items():
            param.data = npz[key]
            
    def forward(self, inputs):
        raise NotImplementedError()
    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
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

class RNN(Layer):
    def __init__(self, hidden_size, in_size = None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size = in_size) #x에서 은닉상태 h로 변환
        self.h2h = Linear(hidden_size, in_size = in_size, nobias = True) #이전 은닉상태에서 다음 은닉상태로 변환
        self.h = None
    def reset_state(self):
        self.h= None
    def forward(self,x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x)+self.h2h(self.h))
            self.h = h_new
        return h_new
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
    
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0,
                 nobias = False, dtype = np.float32, in_channels= None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size= kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        
        self.W = Parameter(None, name = 'W')
        if in_channels is not None:
            self._init_W()
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype = dtype),name = 'b')
    def _init_W(self, xp = np):
        C,OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1/(C*KH*KW))
        W_data = xp.random.randn(OC,C,KH,KW).astype(self.dtype)*scale
        self.W.data = W_data
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y
                    
