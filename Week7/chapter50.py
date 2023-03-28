if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import dezero
from dezero.datasets import Spiral
from dezero import DataLoader,optimizers
import dezero.functions as F
from dezero.models import MLP
max_epoch = 300
batch_size= 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set , batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

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