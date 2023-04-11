from Week1.chaper2 import Square
from chaper1 import Variable
import numpy as np
f = Square()
x = Variable(np.array(2.0))

def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data - y0.data) / (2*eps)
dy=numerical_diff(f,x)
print(dy)