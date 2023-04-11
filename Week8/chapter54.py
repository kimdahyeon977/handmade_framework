if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np
from dezero import test_mode
import dezero.functions as F

x= np.ones(5)
print(x)

y= F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)