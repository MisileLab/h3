알고리즘: 문제를 해결하기 위한 방법

# 좋은 알고리즘의 조건

- 속도가 빨라야 함
  - 상황에 따라 세 가지로 나눔(최악, 보통, 최선)
  - Big O Notation 사용
    - Big O Notation = O(n)식으로 표기하는 것
    - 상대적으로 의미없는 수(최고차항을 제외한 나머지와 최고차항의 계수)를 제외함
- 공간을 적게 차지해야 함(보통 메모리 공간을 의미)

# 선택 정렬

속도: O(n^2)

## 초기 변수

배열의 첫번째 값은 1로 가정, i=1

## 방법

- 배열의 최솟값과 i를 바꾸기
- i를 1 증가
- i가 배열의 길이-1과 같을 때까지 반복 

## 시뮬레이션

./simulations/SelectionSort.mp4