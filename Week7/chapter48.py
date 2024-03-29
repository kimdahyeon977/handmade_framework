if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import dezero
import math
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

max_epoch = 300
batch_size = 30
hidden_size = 10
lr=1.0

x,t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3)) #첫번째 완전 연결계층의 출력크기는 hidden_size, 두번쨰는 3인것
optimizer = optimizers.SGD(lr).setup(model)
data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss=0
    
    for i in range(max_iter):
        batch_index = index[i * batch_size:(i+1)*batch_size]
        batch_x= x[batch_index]
        batch_t = t[batch_index]
        
        y=model(batch_x)
        loss = F.softmax_cross_entropy(y,batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)
    
    avg_loss = sum_loss / data_size
    print(f'epoch {epoch + 1} , loss {avg_loss}')

