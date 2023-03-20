from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L

class Model(Layer): #Layer하위 class
    def plot(self, *inputs, to_file = 'model.png'):
        y=self.forward(*inputs) #forwad로 계산
        return utils.plot_dot_graph(y, verbose= True, to_file = to_file) #생성된 계싼 그래프를 이미지 파일로 내보내기
class MLP(Model):
    def __init__(self, fc_output_sizes, activation= F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i,out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i),layer)
            self.layers.append(layer)
    def forward(self, x):
        for l in self.layers[:-1]:
            x=self.activation(l(x))
        return self.layers[-1](x)
            