
"""
Chapter 2: 데이터 전처리 및 시각화

Minmax Scaling을 사용하기 위한 모듈입니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. minmax_scaling: train 데이터로 min과 max를 계산한 후, train 데이터와 test 데이터에 대해 한 번에 minmax_scaling을 진행합니다.
"""

import numpy as np
import pandas as pd


class Scaling(object): ## object에 train 데이터의 X변수가 입력이 될 예정?
    def __init__(self, train): ## initialization -> 인스턴스가 생성될 때 항상 실행이 되는 함수
        """
        :param
        inputs: train 데이터(X)를 입력으로 받습니다. (형식: pd.DataFrame)
        """

        # 1. 입력받은 train 데이터의 복사본을 만들어 self.X에 저장합니다. (HINT: copy()를 활용하세요.)
        self.X = train.copy() ## train 데이터의 복사본을 self.X에 저장
        self.min = self.X.min(axis = 0) ## 트레인 데이터의 min 데이터 객체 생성
        self.MAX = self.X.max(axis = 0) ## 트레인 데이터의 max 데이터 객체 생성

    def minmax_scaling(self, test): ## test 변수를 담고, 이후에 이를 통해서 함께 minmax scaling 진행 (train 진행한 모듈로, test도 진행하기 위함이다...)
        """
        :param
        test_data: test 데이터(X)를 입력으로 받습니다. (형식: pd.DataFrame)
        :returns Minmax Scaling이 진행된 train 데이터와 test 데이터를 반환합니다.
        """
        test_X = test.copy()
        
        ### train 부터 minmax 진행
        minmax_train = (self.X - self.min)/ (self.MAX - self.min) ## train minmax 적용
        minmax_test = (test_X - self.min)/ (self.MAX - self.min) ## test minmax 적용
        return minmax_train, minmax_test ## 또 return 을 2개로 던져주도록 하자!!
        

        # 1. train 데이터의 속성의 개수를 dim 객체에 저장하고, test 데이터의 복사본을 test1에 저장하세요.
        # 2. train 데이터의 속성별 min과 max를 찾아 train과 test 데이터 모두에 minmax scaling을 진행하세요. (HINT: for문을 사용하세요.)


