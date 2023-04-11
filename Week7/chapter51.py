if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import dezero
from dezero.datasets import MNIST
import numpy as np
from dezero import DataLoader,optimizers
import dezero.functions as F
from dezero.models import MLP

max_epoch = 5
batch_size = 100
hidden_size = 1000

def f(x):
    x= x.flatten()
    x= x.astype(np.float32)
    x/=225.0
    return x

train_set = MNIST(train=True, transform=f)
test_set = MNIST(train=False, transform=f)
train_loader = DataLoader(train_set , batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10),activation=F.relu) #output출력 사이즈 class에 맞게 조정
optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    sum_loss , sum_acc = 0,0
    for x,t in train_loader:
        y= model(x)
        loss = F.softmax_cross_entropy(y,t)
        acc = F.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    print(f'epoch : {epoch+1}')
    print(f'train loss: {sum_loss / len(train_set)} , accuracy : {sum_acc/len(train_set)}')
    
    sum_loss , sum_acc = 0,0
    with dezero.no_grad():
        for x,t in test_loader:
            y= model(x)
            loss = F.softmax_cross_entropy(y,t)
            acc = F.accuracy(y,t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    print(f'test loss: {sum_loss / len(test_set)} , accuracy : {sum_acc/len(test_set)}')
