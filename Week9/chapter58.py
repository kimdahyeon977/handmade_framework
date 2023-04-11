if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero.models import VGG16
import dezero
from PIL import Image

url = 'https://github.com/WegraLee/deep-learning-from-scratch-3/'\
      'raw/images/zebra.jpg'

img_path = dezero.utils.get_file(url)
img= Image.open(img_path)
x= VGG16.preprocess(img)
x= x[np.newaxis]
model = VGG16(pretrained=True)
with dezero.test_mode():
    y=model(x)
predict_id = np.argmax(y.data)
model.plot(x,to_file='vgg.pdf')
labels= dezero.datasets.ImageNet.labels()
print(labels[predict_id])