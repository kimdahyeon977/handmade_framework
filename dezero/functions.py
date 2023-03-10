import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils
# -----
class Sin(Function):
		def forward(self, x):
				y = np.sin(x)
				return y

		def backward(self, gy):
				x, = self.inputs
				gx = gy * cos(x)
				return gx

def sin(x):
	return Sin()(x)

# -----
class Cos(Function):
		def forward(self, x):
				y = np.cos(x)
				return y

		def backward(self, gy):
				x, = self.inputs
				gx = gy * -sin(x)
				return gx

def cos(x):
		return Cos()(x)

#쌍곡탄젠트 or 하이퍼볼릭 탄젠트

class Tanh(Function):
    def forward(self, x):
        y=np.tanh(x)
        return y
    def backward(self, gy):
        y=self.outputs[0]()
        gx= gy * (1-y*y) #tanh의 미분
        return gx
def tanh(x):
    return Tanh()(x)
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Reshape(Function):
    def __init__(self,shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape #역전파에서 입력 x의 형상을 기억해두기
        y= x.reshape(self.shape)
        return y
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

def transpose(x, axes=None):
    return Transpose(axes)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y= np.broadcast_to(x, self.shape)
        return y
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
def broadcast_to(x,shape):
    if x.shape == shape: #broadcast 전 == 후
        return as_variable(x)
    return BroadcastTo(shape)(x)
    
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        self.x_shape = x.shape
        y= x.sum(axis=self.axis, keepdims = self.keepdims)
        return y
    def backward(self, gy):
        #gy의 형상을 '미세하게 조정'한다(axis와 keepdims를 지원하게 되면서 기울기의 형상이 변화하는 경우가 생기기때문!)
        gx = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) #입력 변수와 형상이 같아지도록 기울기 gy의 원소를 복사
        gx = broadcast_to(gy, self.x_shape)
        return gx
def sum(x,axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class SumTo(Function):
    def __init__(self,shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y=utils.sum_to(x,self.shape)
        return y
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape) #역전파에서 입력 x와 형상이 같아지도록 기울기의 원소를 복제
        return gx
def sum_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y= x.dot(W) #ndarray인스턴스에도 대응할 수 있음.
        return y
    def backward(self, gy):
            x,W = self.inputs
            gx = matmul(gy, W.T) #T는 38단계에서 구현한 transpose함수 사용됨.
            gW = matmul(x.T, gy)
            return gx, gW

def matmul(x,W):
			return MatMul()(x,W)