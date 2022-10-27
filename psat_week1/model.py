
"""
Chapter 3: 모델링 - Normal Equation

이후 만들게 될 회귀모형들의 부모 클래스가 되는 Model 클래스를 먼저 만들어 보겠습니다.
Model 클래스는 다음의 3가지 함수를 가지고 있습니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. __init__       : class가 처음 정의될 때 자동으로 생성되는 variable들을 정의합니다.
2. describe       : X와 Y의 기술통계량을 선택적으로 확인합니다.
3. predict        : test 데이터에 대한 모델의 예측값을 반환합니다.
"""

import pandas as pd
import numpy as np


class Model(object):
    def __init__(self, data1, data2):
        """
        :param
        data1: X값들을 입력으로 받습니다. (형식: pd.DataFrame)
        data2: y값들을 입력으로 받습니다. (형식: pd.DataFrame)
        """
        self.X = data1
        self.Y = data2
        self.m = data1.shape[0] ## self.theta에 넣을 array 길이를 위해서 사용
        self.n = data1.shape[1] ## array를 붙이기 위해서 사용 -> 열 크기를 통해서 위치 결정
        self.data1_re = data1.copy() ## 변수를 새로 저장할 하나의 객체를 생성 -> 값을 직접 바꾸게 되면 다시 run하면 오류가 발생
        self.data1_re.insert(self.n, "Additional", 1) ## 맨 오른쪽 항에 1로 된 열을 붙이기
        ### array 로 변경해서 저장
        self.X_mat = np.array(self.data1_re)
        self.Y_mat = np.array(self.Y)
        ## 회귀 계수를 저장하기 위해서 임시로 1로 채워진 array를 self.theta에 저장
        self.theta = np.ones((1, self.m))
        ### 이 방법도 사용 가능
        #### self.theta = np.full((1, self.m), 1)
        ## 빈 객체 생성 -> 파이썬(객체 지향 언어)의 장점이라 할 수 있다. 어떤 형식이든지 담을 수 있다. -> 평소에 변수 수정을 진행할 때도, type이 지정되어있는 것이 아닌 원래는 이러한 객체 형식이기 때문에 어떤 타입이든지 지정할 수 있는 것이다.
        self.train_pred = object()
        self.test_pred = object()
        
    # 1. data1의 row 개수를 self.m, data1의 col 개수를 self.n에 각각 저장하세요.
    # 2. data1의 오른쪽에 1로 채워진 열을 붙여 array로 바꾼 후 self.X_mat에 새로 저장하세요.
    # 3. self.Y를 array로 바꾼 후 self.Y_mat에 새로 저장하세요.
    # 4. 회귀계수를 저장하기 위해 임시로 1로 채워진 array를 self.theta에 저장하세요.
    # 5. 예측값을 저장하기 위한 빈 객체로 self.train_pred와 self.test_pred를 생성하세요.
    ####################################### 완료 #################################################

    def describe(self, which):
        """
        :param
        which: 'X' 혹은 'y'를 입력으로 받습니다. (형식: str)
        :return: pd.DataFrame.info()
        """
        
        # 1. if문을 활용하여 'X'가 입력되었을 때는 data1의 info를, 'y'가 입력되었을 때는 data2의 info를 반환하도록 하세요.
        # 2. 만약 2가지 이외의 무언가가 입력되었을 때는 "Not Defined"라는 예외를 띄우도록 하세요. (HINT: raise Exception을 사용하세요.)
        
        if (which == "X"): ## X를 입력받았을 경우에, data1을 담고 있는 self.X에 info()를 적용
            return self.X.info()
        elif (which == "y"): ## y를 입력받았을 경우에, data2을 담고 있는 self.y에 info()를 적용
            return self.Y.info()
        else: ## 이외가 입력이 되었을 경우에, Not Defined 예외 출력 -> raise Exception을 사용하면, 예외 에러 코드 사용 가능
            raise Exception("Not Defined")
        
        



    def predict(self, test1):
        """
        :param
        test1: test 데이터를 입력으로 받습니다.        (형식: pd.DataFrame)
        :return: test 데이터에 대한 예측 결과를 반환합니다. (형식: list)
        """

        # 1. test 데이터에 self.X_mat과 동일한 처리를 진행하여 test_data 객체에 저장하세요.
        # 2. test_data와 self.theta를 곱해서 test 데이터에 대한 예측값을 출력하세요. (HINT: np.matmul을 사용하세요.)
        
#         test = test1
#         test_m = test.shape[0]
#         test_n = test.shape[1]
#         test.insert(test_n, "Additional", 1) ## 맨 오른쪽 항에 1로 된 열을 붙이기
#         test_data = np.array(test)

        self.test = test1 ## 원본 데이터 담아놓기
        self.length_col = self.test.shape[1]
        self.test1_re = test1.copy()
        self.test1_re.insert(self.length_col, "Additional", 1) ## train과 똑같은 처리 시작
        test_data = np.array(self.test1_re) ## array로 변형
        
        ##### test_data, self.theta 곱해서 test 데이터에 대한 예측값 출력
        res = np.matmul(test_data, self.theta)
        self.test_pred = list(pd.DataFrame(res.tolist())[0])
        return self.test_pred
        
        

