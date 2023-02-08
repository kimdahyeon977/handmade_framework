#역전파 자동화#
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data 
        self.grade = None #미분값이 들어갈것임
        self.creator = None
    def set_creator(self, func): #creator를 설정할 수 있도록 메서드 추가
        self.creator = func

class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        output.set_creator(self) #출력 변수에 창조자를 설정한다.
        self.input = input #self를 쓴 이유 : 입력 변수를 기억
        self.output = output #출력도 저장
        return output

#코드 실행#
