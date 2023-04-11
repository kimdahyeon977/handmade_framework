#Variable 클래스 구현
class Variable:
    def __init__(self, data):
        self.data=data

import numpy as np
data=np.array(1.0)
x=Variable(data)
print(x) #<__main__.Variable object at 0x000001AF7B2E5D60>
print(x.data) #1.0



