
if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가
from dezero import Model
import dezero.functions as F
import dezero.layers as L
import numpy as np
import dezero.optimizers as op
import dezero
max_epoch = 100
hidden_size = 100
bptt_length = 30
train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)


class SimpleRNN(Model):
		def __init__(self, hidden_size, out_size):
					super().__init__()
					self.rnn = L.RNN(hidden_size)
					self.fc = L.Linear(out_size)
		def reset_state(self):
					self.rnn.reset_state() #은닉 상태를 재설정
		def forward(self, x):
					h= self.rnn(x)
					y= self.fc(h)
					return y

model = SimpleRNN(hidden_size,1)
optimizer = op.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0,0
    for x,t in train_set:
        x= x.reshape(1,1)
        y= model(x)
        loss += F.mean_squared_error(y,t)
        count +=1
        if count % bptt_length ==0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    avg_loss = float(loss.data)/ count
    print(f' epoch : {epoch +1} , loss : {avg_loss}')