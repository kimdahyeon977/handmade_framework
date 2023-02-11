from .utils import *
import numpy as np
'''
여러개의 input과 output으로 변경해보기 => 리스트(튜플)에 넣어서 처리
'''
class Function:
    def __call__(self, inputs):
        xs= [x.data for x in inputs]
        ys=self.forward(xs)
        outputs=[Variable(np.asarray(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs
        
    def forward(self, x):
        raise NotImplementedError()
    def backwrad(self, gy):
        raise NotImplementedError()

#코드 실행
xs=[Variable(np.array(2)), Variable(np.array(3))] #리스트로 준비
f=Add()
ys=f(xs) #(output, )
y=ys[0]
print(y.data)

        