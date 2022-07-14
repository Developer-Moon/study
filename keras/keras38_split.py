
import numpy as np   

a = np.array(range(1, 11))                     # a = [ 1  2  3  4  5  6  7  8  9 10]
size = 7                                       # 범위 밖의 숫자들은 에라   size >= 0, size <= 10

def split_x(dataset, size):                    # split_x 함수는 매개변수 dataset, size가지고 있다
    aaa = []                                   # aaa = [] 빈 리스트 선언
    for i in range(len(dataset) - size + 1):   # i라는 변수를 사용하여 for문을 사용 range(10 - size + 1) = range(6)     len(dataset) 그냥 10이라고 해주시지... 너무하시네
        subset = dataset[i : (i + size)]       # size=5일때 a = [1 2 3 4 5 6] 이라서 subset은 [1:6]=1 2 3 4 5, [2:7]=2 3 4 5 6 ....[6:11]=6 7 8 9 10
        aaa.append(subset)                     # 변수 aaa에 append함수를 사용하여 subset을 더해준다 
    return np.array(aaa)                       # 그 return값이 np.array(aaa)

bbb = split_x(a, size)                         # 변수 bbb = split함수에 a와 size라는 인수를 적용한 함수다
print(bbb)
print(bbb.shape)        #(6, 5)

x = bbb[:, :-1]                                # x는 [행, 열] 행은 전체, 열은 뒤에서 두번째까지 나타낸다
y = bbb[:, -1]                                 # 열에서 뺸다(여기서 -1은 인덱스 뒤에서부터)
print(y)
print(x.shape, y.shape) #(6, 4)(6.)

import numpy as np   


a = np.array(range(1, 11))                     # a = [ 1  2  3  4  5  6  7  8  9 10]
size = 7                                       # 범위 밖의 숫자들은 에라   size >= 0, size <= 10

def split_x(dataset, size):                    # split_x 함수는 매개변수 dataset, size가지고 있다
    aaa = []                                   # aaa = [] 빈 리스트 선언
    for i in range(len(dataset) - size + 1):   # i라는 변수를 사용하여 for문을 사용 range(10 - size + 1) = range(6)     len(dataset) 그냥 10이라고 해주시지... 너무하시네
        subset = dataset[i : (i + size)]       # size=5일때 a = [1 2 3 4 5 6] 이라서 subset은 [1:6]=1 2 3 4 5, [2:7]=2 3 4 5 6 ....[6:11]=6 7 8 9 10
        aaa.append(subset)                     # 변수 aaa에 append함수를 사용하여 subset을 더해준다 
    return np.array(aaa)                       # 그 return값이 np.array(aaa)

bbb = split_x(a, size)                         # 변수 bbb = split함수에 a와 size라는 인수를 적용한 함수다
print(bbb)
print(bbb.shape)        #(6, 5)

x = bbb[:, :-1]                                # x는 [행, 열] 행은 전체, 열은 뒤에서 두번째까지 나타낸다
y = bbb[:, -1]                                 # 열에서 뺸다(여기서 -1은 인덱스 뒤에서부터)
print(y)
print(x.shape, y.shape) #(6, 4)(6.)

