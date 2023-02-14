import numpy as np
class Variable:
    def __init__(self, data, name = None):
        #타입 에러 방지
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported!')
        self.data = data
        self.name = name  #변수의 이름을 붙여줄 수 있다.
        self.grad = None #미분값이 들어갈것임
        self.creator = None #함수를 가져온다.
        self.generation = 0 #세대 수를 기록하는 변수
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation +1 #세대를 기록한다(부모 세대 + 1)
    def backward(self,retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()
        def add_func(f): #DeZero 함수 리스트를 세대 순으로 정렬
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)
        while funcs:
            f= funcs.pop()
            gys = [output().grad for output in f.outputs] #outputs에 담긴 미분값들 리스트에 담기
            gxs = f.backward(*gys) #역전파 호출
            if not isinstance(gxs, tuple): #튜플로 변경
                gxs=(gxs, )
            for x,gx in zip(f.inputs, gxs): #역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장해둔다.
                if x.grad is None:
                    x.grad = gx #처음에는 그대로 대입
                else:
                    x.grad = x.grad + gx #그 다음번부터는 전달된 미분값을 더해줘야함.
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None #y는 약한 참조
    @property #메서드를 인스턴스 변수처럼 사용할 수있다.
    def shape2(self):
        return self.data.shape
    @property
    def ndim(self): #차원 수
        return self.data.ndim
    @property
    def size(self): #원소 수
        return self.data.size
    @property
    def dtype(self): #데이터 타입
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n', '\n'+' '*9) #좀더 보기편하게!
        return 'variable(' + p + ')'
#코드 실행
x=Variable(np.array([[1,2,3,4],[4,5,6,7]]))
print(x)