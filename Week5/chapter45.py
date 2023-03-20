import dezero.layers as L
import dezero.functions as F
from dezero import Layer
import numpy as np
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F

np.random.seed(0)
x= np.random.rand(100,1)
y= np.sin(2*np.pi*x) + np.random.rand(100,1)
lr= 0.2
max_iter= 10000
hidden_size=10

##step2 : Model 클래스는 마치 Layer클래스 처럼 활용할 수 있다.
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        
    def forward(self, x):
        y= F.sigmoid(self.l1(x))
        y= self.l2(y)
        return y
model = TwoLayerNet(hidden_size,1)


#학습 시작
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y,y_pred)
    
    model.cleargrads()
    loss.backward()
    
    for p in model.params():
        p.data -= lr*p.grad_data
    if i%1000==0:
        print(loss)