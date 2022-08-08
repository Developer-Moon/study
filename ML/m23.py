 

from typing import List, Tuple

# 방법2. 정렬해서 최소값을 찾은 후 인덱스 구하기
def method_2(L: List[float]) -> Tuple[int,int]:

    # L의 복사본을 정렬한다
    sorted_list = sorted(L)

    # 가장 작은 두 수를 구한다
    smallest = sorted_list[0]
    next_smallest = sorted_list[1]

    # 원래 리스트 L에서 두 수의 인덱스를 구한다.
    min1 = L.index(smallest)
    min2 = L.index(next_smallest)

    # 두 인덱스를 반환한다
    return (min1, min2)

items = [809, 834, 477, 478, 307, 122, 96, 102, 324, 476]
result = method_2(items)
print(result)


items = [809, 834, 477, 478, 307, 122, 96, 102, 324, 476]
print("%.2f" % (items * 25.0 / 100.0))