
"""
Chapter 3: 모델링 - Normal Equation

행렬 연산을 통한 최소제곱법을 구현하기 위해 Normal Equation의 연산 과정을 모듈로 만들어 보겠습니다.
Model 클래스를 상속받아 사용하겠습니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. normal_eq       : Model 클래스를 통해 입력받은 데이터에 대해 회귀모형을 적합하고, 예측값과 회귀계수를 반환합니다.
"""

import numpy as np
from model import Model
import pandas as pd

class LSE(Model): ### 클래스의 부모 클래스로 Model 사용을 위함 -> 이를 상속이라고 한다.
    # 1. Model 클래스가 LSE 클래스의 부모 클래스가 될 수 있도록 코드를 작성해 주세요. (HINT: 괄호 안에 코드를 작성하세요.)
    def __init__(self ,X_mat ,Y_mat):
        super().__init__(X_mat, Y_mat) 
        # super().__init__()이라는 코드가 다른 클래스의 속성 및 메소드를 자동으로 불러와 해당 클래스에서도 사용이 가능하도록 도와준다.


    def normal_eq(self):
        """
        :return: self.theta, self.train_pred
        """
        # 2. X의 전치행렬과 X를 곱한 후 그 역행렬을 구하여 inverse 객체에 저장하세요 (HINT: np.matmul, np.linalg.inv를 사용하세요.)
        ## 전치행렬과 기본 행렬의 곱
        transpose = np.matmul((self.X_mat).T,self.X_mat)
        ## 역행렬을 구하기 -> inverse에 담기
        inverse_mat = np.linalg.inv(transpose)
        ## 역행렬과 x의 전치 행렬 곱하기
        semi = np.matmul(inverse_mat, (self.X_mat).T)
        ### Y와 곱해서 회귀 계수를 구하기
        self.theta = np.matmul(semi, self.Y_mat)
        self.theta = list(pd.DataFrame(self.theta.tolist())[0])
        # 4. X와 회귀계수를 곱하여 얻은 예측값을 self.train_pred 객체에 저장
        self.train_pred = np.matmul(self.X_mat, self.theta)
        self.train_pred = list(pd.DataFrame(self.train_pred.tolist())[0])
        ## return으로 던져주기
        return self.theta, self.train_pred

        
        
        
        
        
        
        
        