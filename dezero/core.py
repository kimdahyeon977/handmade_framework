import weakref
import numpy as np
import contextlib
import dezero


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
def test_mode():
    return using_config('train',False)

def no_grad():
    return using_config('enable_backprop', False)


# =============================================================================
# Variable / Function
# =============================================================================
try:
    import cupy
    array_types = (np.ndarray , cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)
    
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    #가변 인수를 받는 reshape메서드를 추가
    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data) #gpu to cpu
    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data) #cpu to gpu
            
    def reshape(self, *shape):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    def transpose(self, *axes):
        if len(axes)==0:
            axes = None
        elif len(axes)==1:
            if isinstance(axes[0],(tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)
    def sum(self, axis= None, keepdims = False):
            return dezero.functions.sum(self, axis, keepdims)
    def unchain(self):
        self.creator= None #부모 함수로의 연결을 끊기
    def unchain_backward(self):
        if self.creator is not None:
                funcs = [self.creator]
                while funcs:
                        f = funcs.pop()
                        for x in f.inputs:
                                if x.creator is not None:
                                        funcs.append(x.creator)
                                        x.unchain()    
    @property
    def T(self):
        return dezero.functions.transpose(self)
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False , create_graph = False):
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys) #메인 backward
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx #이계산도 대상

                    if x.creator is not None:
                        add_func(x.creator)

                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None  # y is weakref


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x, array_module = np):
    if np.isscalar(x):
        return array_module.array(x)
    return x

def add(x0,x1):
    x1 = as_array(x1,dezero.cuda.get_array_module(x0.data))
    return Add()(x0,x1)
def mul(x0,x1):
    x1= as_array(x1, dezero.cuda.get_array_moduel(x0.data))
    return Mul()(x0,x1)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 사칙연산 / 연산자 오버로드
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        #broadcast_to 함수의 역전파는 sum_to 함수에 해당한다.
        gx0,gx1 =  gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0,self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0,gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        #x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0,x1 = self.inputs #Variable 인스턴스를 그대로 사용
        gx0 = gy*x1
        gx1 = gy*x0
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0,self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)

class Parameter(Variable):
    pass

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item
    
    Variable.max = dezero.functions.max
    Variable.min = dezero.functions.min