
"""
Chapter 2: 데이터 전처리 및 시각화

Outlier 제거를 위한 모듈입니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. del_outlier: 변수별로 outlier의 index를 추출한 후, 해당 데이터들을 제거합니다.
"""

import pandas as pd

# list_outlier = list() ## 아웃라이어 인덱스를 담을 변수를 설정
global list_outlier ## 전역 변수 설정
list_outlier = list() ## 전역변수에 대한 변수 선언

def is_outlier(data): ## 데이터를 던져주면 처리하도록 합시다!! -> data, y 가 두번 들어오면 되겠죠
    global list_outlier
    ### IQR을 통해서 이상치를 구할 예정입니다. -> 이를 통해서 함수에서 식을 계산하면 편합니다.
    q3 = data.quantile(0.75) ## 3분위
    q1 = data.quantile(0.25) ## 1분위
    iqr = q3 - q1 ## iqr 구하기
    
    ### 반복문을 통해서 outlier의 인덱스를 담도록 하겠습니다.
    for var in data.columns: ## variable 반복문으로 모두 넣기
        target = data[var] ## 이를 통해서 target을 설정한다.
        ## 반복문 두번 쓰는 방식으로 짜보기 -> apply 사용해서 짜면, 조금 더 시간복잡도 측면에서 효율적일듯?
        for i in range(len(data)): ## 데이터 길이만큼 담기
            try:
                if target[i] > q3[var]+ 1.5 * iqr[var] or target[i] < q1[var]- 1.5 * iqr[var]: ## 이상치를 양쪽 범위에서 탐색
                    list_outlier.append(i) ## 전역변수인 리스트에 담는다.
            except: 
                pass
    

        
### data는 Strength 가 빠진 값이 들어올거고, y에는 Strength만 들어올거임
def del_outlier(data, y):
    global list_outlier ## 전역변수를 import 해온다.
    """
    :param
    data   : train 데이터의 X값 (형식: pd.DataFrame)
    y      : train 데이터의 y값 (형식: pd.DataFrame)
    :return: Outlier가 모두 제거된 x값과 y값 (형식: pd.DataFrame)
    
    
    1. for문을 사용하여 변수별로 outlier의 index를 저장한 후, 중복되는 행들을 처리하여 최종적으로 이상치를 제거하세요.
    (HINT: for문을 사용하고, 중복되는 행들을 처리하고 위해 set()을 사용하세요.
    """
    

#     ### x 변수에서 outlier 인덱스 추출
#     is_outlier(data)
#     ### y 변수에서 outlier 인덱스 추출
#     is_outlier(y)

    for input in (data, y): ## 반복문으로 두번 넣어주기
        is_outlier(input)
    
    ### outlier list set을 통해서 겹치는 것 제거 -> 집합을 사용하면 제거가 가능하다. -> 리스트를 통해서 drop 사용 위해서 list로 한번 더 변환
    ### 이를 통해서 전체 인덱스에서 제거하도록 하겠다.
#     print(list_outlier)
    outlier_index = list(set(list_outlier)) ## set 통해서 중복값 제거 이후에 list 로 변환
    
    ## drop을 통해서 열을 제거한다. -> 데이터 프레임으로 던져달라고 했음
    none_outlier_x = pd.DataFrame(data.drop(outlier_index), columns = data.columns) ## drop을 통해서 이상치 버리기 drop의 default axis = 0 -> 행을 버리게 된다.
    none_outlier_y = pd.DataFrame(y.drop(outlier_index), columns = y.columns) ## 위와 동일
    
    ## x,y 따로 해서 return 던져주면 되겠죠??
    return none_outlier_x, none_outlier_y ## 이렇게 던져주면. 함수를 통해서 변수 두개를 받아줄 수 있음

    """
        이렇게 밖에서 사용하도록 하자!!
        df_x, df_y = del_outlier(data, y)
    """
    
    
    


            
        

            
    
    
    
    
