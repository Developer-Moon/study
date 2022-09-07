gradient_01 = lambda x : 2*x - 4  # 함수화를 간편하게 할 수 있다

def gradient_02(x) :
    temp = 2*x - 4
    return temp

x = 3

print(gradient_01(x))
print(gradient_02(x))