if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
import dezero.functions_conv as F
from dezero import Variable
x1 = np.random.rand(1,3,7,7)
col = F.im2col(x1, kernel_size=5, stride=1, pad =0, to_matrix=True)
print(col.shape)

x2= np.random.rand(10,3,7,7)
kernel_size=(5,5)
stride = (1,1)
pad = (0,0)
col2 = F.im2col(x2, kernel_size, stride , pad, to_matrix = True)
print(col2.shape)

N,C,H,W = 1,5,15,15
OC, (KH,KW)= 8,(3,3)

x= Variable(np.random.randn(N,C,H,W))
W = np.random.randn(OC,C,KH,KW)
y= F.conv2d_simple(x,W,b=None, stride = 1, pad=1)
y.backward()

print(y.shape)
print(x.grad.shape)