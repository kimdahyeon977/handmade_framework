#파이썬 명령어를 어디에서 실행하든 dezero 디렉터리의 파일들은 제대로 임포트 할수있게
if '__file__' in globals(): #__file__이라는 전역변수가 정의되어있는지 확인
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..')) #이 파일의 부모 디렉터리를 모듈 검색 경로에 추가


import numpy as np
from dezero import Variable

x=Variable(np.array(1.0))
print(x)