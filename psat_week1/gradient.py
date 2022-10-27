
"""
Chapter 4: 모델링 - Gradient Descent

경사하강법을 통해 회귀모형을 적합할 수 있는 모듈을 만들어 보겠습니다.
Model 클래스를 상속받아 사용하겠습니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. __init__       : gradient를 계산할 parameter들을 지정하고, 모든 값들을 tensor로 변경합니다.
2. compute_cost   : gradient 계산에 활용되는 손실함수를 계산합니다.
3. comput_gradient: 주어진 반복횟수와 학습률에 따라 회귀계수를 추정합니다.
"""

import torch
import time
import pandas as pd
from model import Model
import numpy as np


class GradientDescent(Model):
    def __init__(self, data1, data2):
        super().__init__(data1, data2)
        # super().__init__()이라는 코드가 다른 클래스의 속성 및 메소드를 자동으로 불러와 해당 클래스에서도 사용이 가능하도록 도와준다.
        """
        :param
        data1   : Model 클래스와 동일
        data2   : Model 클래스와 동일
        """

        # 1. Model 클래스가 LSE 클래스의 부모 클래스가 될 수 있도록 코드를 작성해 주세요. (HINT: 20번째 줄 괄호 안에 코드를 작성하세요.) -> 완료
        # 2. super()를 통해 Model 클래스의 method와 입력들을 GradientDescent로 가져오세요. (HINT: super().__init__()을 사용하세요.) -> 완료

        # 3. 필요한 회귀계수의 수만큼 self.theta_i(ex) self.theta_1)와 self.bias를 객체로 만들고, gradient가 흐를 수 있는 tensor를 저장해 주세요.
        #    (HINT: torch.ones()를 사용하고, requires_grad 인자를 알아보세요.)  
        self.theta_1 = torch.ones(1, requires_grad=True) ## 나중에 tensor을 담을 빈 tensor를 제작하는 과정이다. 각 열 별로 저장할 수 있도록 객체를 생성하였다.
        self.theta_2 = torch.ones(1, requires_grad=True)
        self.theta_3 = torch.ones(1, requires_grad=True)
        self.theta_4 = torch.ones(1, requires_grad=True)
        self.theta_5 = torch.ones(1, requires_grad=True)
        self.bias = torch.ones(1, requires_grad=True)

        # 4. Model 클래스의 self.X_mat의 모든 열들을 분리하여 각 tensor로 저장하세요. (HINT: torch.as_tensor()를 사용하세요.)
        self.tensor_X = torch.as_tensor(self.X_mat.T) ## 각 열별로 쪼개기 위해서 Transpose를 진행한 후, torch.as_tensor를 진행했다.


        # 5. 회귀계수의 변화 과정을 저장하기 위해 각 회귀계수의 이름(theta_1, bias 등)을 key로 하고 빈 list를 value로 갖는 dictionary를 생성하세요. -> iterator를 돌면서 작성될거임
        self.theta_hist = {
            'theta_1' : list(),
            'theta_2' : list(),
            'theta_3' : list(),
            'theta_4' : list(),
            'theta_5' : list(),
            'bias' : list()
        }

        # 6. 손실함수 값의 변화 과정을 저장하기 위해 self.cost_hist에 빈 list를 저장하세요. 그리고 self.cost에 0을 저장하여 초기화하세요.
        self.cost_hist = list() ## 빈 리스트 생성
        self.cost = 0

        # 7. 예측값을 저장하기 위한 빈 객체로 self.pred를 생성하세요.
        self.pred = object() ## 빈 객체 생성
    
    def compute_cost(self):
        """
        :return: MSE 값을 반환합니다.
        """

        ### y 데이터 처리
        col = self.Y.columns[0] ## 컬럼 이름 뽑기
        Y_list = list(self.Y[col])
        # 1. self.pred와 실제값 self.y를 사용하여 MSE를 계산하고, self.cost에 저장하여 반환하세요.
        self.cost = sum(([(x-y)** 2 for x,y in zip(Y_list , self.pred.tolist())]) )/ len(Y_list) ## MSE 계산식 -> zip을 통해서 리스트의 차를 구하는 함수를 사용, 잔차 제곱의 평균을 구했다. -> 제곱은 **로 써야 한다!! 주의!!
        return self.cost
        

    def compute_gradient(self, num_iter, lr, optimizer):
        """
        :param
        num_iter   : 반복횟수를 지정합니다.   (형식: int)
        lr         : 학습률을 지정합니다.     (형식: float)
        optimizer  : optimizer를 지정합니다. (형식: torch.optim)
        :return: self.theta, self.cost_hist, self.theta_hist, self.train_pred
        """
        
        # 1. 입력받은 optimizer로 회귀계수를 업데이트하기 위한 optimizer 객체를 생성하세요. (HINT: optimizer(회귀계수 목록, lr=학습률) 형태입니다.
        
        ### 회귀계수 목록 선언
        list_tensor = [self.theta_1, self.theta_2, self.theta_3, self.theta_4, self.theta_5, self.bias] ## 위에 존제하는 각 회귀계수의 tensor를 넣어준다.
        ## optimizer 객체 생성
        optimizer = optimizer(list_tensor, lr = lr) ## 최적화 함수
        # 2. 모형 적합에 소요되는 시간을 측정하기 위해 start 객체에 현재 시간을 저장하세요.
        start = time.time() ##시작 시간을 넣는다.
        
        # 3. num_iter만큼 for문을 반복하여 회귀계수를 적합하세요.
        for epoch in range(num_iter + 1): ## 마지막까지 돌기 위해서 +1 돌았음 -> 0에서 시작, num_iter까지 돌기
            # 3-1. 각 theta와 대응되는 X의 열들을 곱하여 self.pred에 저장하세요. -> 각 theta의 텐서값과, x의 텐서 값의 곱을 통해서 pred를 구한다. 
            self.pred =  self.tensor_X[0]* self.theta_1+  self.tensor_X[1]* self.theta_2 + self.tensor_X[2]*self.theta_3 + self.tensor_X[3]*self.theta_4 + self.tensor_X[4]*self.theta_5 + self.tensor_X[5]*self.bias
            
            # 3-2. cost를 계산하고, self.cost_hist에 저장하도록 하세요.
            cost = torch.mean((self.pred - torch.as_tensor(self.Y_mat.T)) ** 2) ## 손실함수 계산: 먼저 cost를 작성한 이후에,
            self.cost_hist.append(cost.tolist()) ## cost_hist에 이 cost를 담아서 list 형태로 제작해주었다.
              
            
            # 3-3. self.theta_hist의 각 key에 대응되는 theta값들을 저장하세요. (HINT: tensor.clone().detach().numpy()로 저장하는 것이 안전합니다.)
            ## 이게 tensor 가 신기한게, optimizer로 경사하강법을 진행하면 그래프로 표현한다고 합니다 그래서 detach를 사용해야 하며, 그 이후에 numpy를 사용해서 배열로 저장할 수 있다고 합니다.
            self.theta_hist["theta_1"].append(self.theta_1.clone().detach().numpy().tolist()[0]) ## 숫자만 넣어주기 위해서 list 로 변환 이후에, index == 0 인 값으로 불러와줬다. -> theta_hist 는 결국 리스트로 작성이 되었다.
            self.theta_hist["theta_2"].append(self.theta_2.clone().detach().numpy().tolist()[0])
            self.theta_hist["theta_3"].append(self.theta_3.clone().detach().numpy().tolist()[0])
            self.theta_hist["theta_4"].append(self.theta_4.clone().detach().numpy().tolist()[0])
            self.theta_hist["theta_5"].append(self.theta_5.clone().detach().numpy().tolist()[0])
            self.theta_hist["bias"].append(self.bias.clone().detach().numpy().tolist()[0])

            # 3-4. cost를 역전파시키고, optimizer가 알맞은 회귀계수를 추정해나갈 수 있도록 하세요.
            optimizer.zero_grad() ## optimizer 초기화
            cost.backward() ## cost 역전파
            optimizer.step() ## optimizer 진행
            end = time.time() ## 시작시간 작성
            # 3-5. 100번의 Epoch마다 소요된 시간과 손실함수 값을 반환하게 하세요.
            if epoch % 100 == 0:
                print('Epoch {:4d}/{} time: {} Cost: {:.6f}'.format(
                    epoch, num_iter, (end - start) , cost.item()))
        # 4. self.train_pred에 for문의 마지막 self.pred값을 저장하세요. (HINT: tensor.detach().numpy()를 사용하세요.)
        self.train_pred = self.pred.detach().numpy()
        # 5. self.theta에 최종 theta 값을 list 형태로 저장하세요. (HINT: 각 theta 값에 tensor.detach().numpy()를 사용하세요.)
        self.theta = []
        for theta_num in ("theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "bias"): ## 반복문으로 쉽게 담아주기
            self.theta.append(self.theta_hist[theta_num][-1]) ### 마지막 theta_hist 값을 theta에 저장한다. -> 인덱스를 -1로 줌에 따라서 마지막 결과 물을 담게 된다.
        return self.theta, self.cost_hist, self.theta_hist, self.train_pred
