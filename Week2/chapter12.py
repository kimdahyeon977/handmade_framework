from utils import Variable
import numpy as np
class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs) #self.forward(x0, x1)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[Variable(np.asarray(y) for y in ys)] #반환 원소가 하나뿐이라면 해당 원소를 직접 반환
        
        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs
        self.outputs= outputs
        # 리스트의 원소가 한개라면 첫번째 원소를 반환
        print(self.inputs)
        return outputs if len(outputs) >1 else outputs[0]
class Add(Function): #Function의 모든 인스턴스들을 상속받는다.
    def forward(self, x0, x1):
        y=x0+x1
        return y
def add(x0, x1):
    return Add()(x0,x1)

#코드 실행
x0=Variable(np.array(2))
x1=Variable(np.array(3))
f=Add()
y=add(x0,x1)
print(y.data)
