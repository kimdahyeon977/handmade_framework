import math
import numpy as np
#매개변수 갱신을 위한 클래스
class Optimizer:
    def __init__(self):
        self.target = None #두 인스턴스를 초기화
        self.hooks = []
    def setup(self, target):
        self.target = target #매개변수를 가지는 클래스를 인스턴스 변수인 target으로 설정
        return self
    def update(self):
        #none이외의 매개변수를 리스트에 모아둠
        params = [p for p in self.target.params() if p.grad is not None]
        #전처리
        for f in self.hooks:
            f(params)
        for param in params:
            self.update_one(param)
    def update_one(self, param):
        raise NotImplementedError()
    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer): #Optimizer 클래스 상속
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    def update_one(self,param): #매개변수 갱신을 SGD에게 맡길 수 있다.
        param.data -= self.lr *param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr =0.01, momentum =0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {} #계속 update할 속도에 해당하는 매개변수
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        v= self.vs[v_key]
        v*= self.momentum
        v-= self.lr * param.grad.data
        param.data += v
        
class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):

        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (np.sqrt(v) + eps)