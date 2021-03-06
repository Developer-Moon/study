# 클래스(Class): 반복되는 불필요한 소스코드를 최소화 하면서 현실 세계의 사물을
#              프로그램 상에서 쉽게 표현할 수 있도록 해주는 프로그래밍 기술
# 인스턴스: 클래스로 정의된 객체를 프로그램 상에서 이용할 수 있게 만든 변수

# 클래스의 맴버: "클래스 내부에 포함되는 변수"
# 클래스의 함수: "클래스 내부에 포함되는 함수"(메소드라고 부르기도 한다. ) 

class Car:
    # 클래스의 생성자     __init을 사용하며 ()안에 'self'라는 메소드를 가지도 있다. 
    def __init__(self, name, color):
        self.name = name     # 클래스의 맴버 (인스턴스 안에 포함된 변수이므로 맴버라고 부른다.)
        self.color = color   # 클래스의 맴버 (인스턴스 안에 포함된 변수이므로 맴버라고 부른다.)
        
     #클래스의 메소드
    def show_info(self): 
        print('이름:', self.name, "/ 색상:", self.color)
        
    # Setter 메소드
    def set_name(self, name):  # 특정한 속성의 값을 변경할 때 사용한다.
        self.name = name  
        
    # 클래스 소멸자 (인스터스가 소멸되었을 때 처리해주는 함수)   
    def __del__(self):
        print("인스턴스를 소멸시킵니다.")  
        
###############################################################(하나의 클래스 완성)

     # 인스턴스 생성 (클래스를 불러와서 사용하며 인스턴스는 사용자가 원하는 만큼 생성 가능)
# 인스터스 1번
car1 = Car("소나타", "빨간색")
car1.show_info()

# 인스터스 2번
car2 = Car("아반떼", "검은색")
car2.show_info()

# 클래스 소멸자를 이용해서 인스터스 2번을 소멸하기
del car2

     #이처럼 사용자가 원하는 자료구조를 만들기 위해서 클래스를 사용한다(이번 인스터스에서 만든 것은 자동차형 자료구조를 생성한 것.)
    
print(car1.name, "을(를) 구매했습니다. ")  # .을 이용하여 맴버 변수에 접근할 수 있다.

#################################################################(인스터스 완성)

      # Setter 메소드 사용
        
car1 = Car("소나타", "빨간색")
car1.set_name("아반떼")   # 소나타를 아반떼로 속성값을 변경함
print(car1.name, "을(를) 구매했습니다. ")
