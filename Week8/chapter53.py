import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
import os
max_epoch = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000,10))
optimizer = optimizers.SGD().setup(model)

#매개변수 읽기
if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')
#GPU모드
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0
    
    for x,t in train_loader:
        y= model(x)
        loss = F.softmax_cross_entropy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
    
    elapsed_time = time.time() - start
    print(f'eopch : {epoch+1} , loss : {sum_loss / len(train_set)} , time : {elapsed_time}')
    
    model.save_weights('my_mlp.npz')