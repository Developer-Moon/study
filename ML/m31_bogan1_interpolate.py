#   결측지 처리

#1. 행 또는 열 삭제[무조건 삭제하는건 무식한 방법]

#2. 임의의 값 : 
#   평균 - maen
#   중위 - median
#   0    - fillna(0)
#   앞값 - ffill
#   뒷값 - bfill
#   특정값
#   기타등등
#   평균과, 중위값(중간값)은 통상적으로 잘 넣지 않는다 [중위값은 생각없이 넣을때]

#3. 보간 - interpolate(선형 형식, linear방식으로 찾아낸다)

#4. 모델을 만든 후 - predict [결측지 를 predict에 넣고 결과값을 결측지에 채운다] 모델은 사용하고 싶은거 사용
# 부스팅계열 - 이상치에 대해 자유롭다 믿거나 말거나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 

from datetime import datetime
import pandas as pd1
import numpy as np

dates = ['8/10/2022', '8/11/2022', '8/12/2022', '8/13/2022', '8/14/2022']

dates = pd.to_datetime(dates)
print(dates)

ts = pd.Series([2, np.nan, np.nan, 8, 10 ], index=dates) # Series 컬럼 하나 형태
print(ts)

print('______________________________')

ts = ts.interpolate()  # linear형태로 결측지가 채워진다
print(ts)