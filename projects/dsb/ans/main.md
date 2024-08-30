알고리즘: 문제를 해결하기 위한 방법

# 좋은 알고리즘의 조건

- 속도가 빨라야 함
  - 상황에 따라 세 가지로 나눔(최악(Big O), 보통(omega), 최선(세타))
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

[SelectionSort Simulation](https://raw.githubusercontent.com/MisileLab/h3/main/projects/dsb/ans/simulations/SelectionSort.mp4)

# 삽입 정렬

속도: O(n^2)

## 안정성이 있는 이유

안정성: 동일한 값의 순서가 변경되지 않는것
동일한 값이 왼쪽에 있을 때 비교할 시, 비교는 하지만 이동하지 않고, 오른쪽에 있을 때는 비교하지 않음

## 방법

- 리스트 안에 정렬할 값을 n이라고 가정
- 배열 중 적절한 자리(왼쪽 수가 n보다 작고 오른쪽 수가 n보다 큰 자리)에 배치

## 성능

- 가장 빠르게 정렬되는 경우
  - 모두 정렬되어 있을 때
  - 총 비교 횟수: n-1
  - 이동 횟수: 0
- 가장 느리게 정렬되는 경우
  - 역순으로 정렬되어 있을 때
  - 총 비교 횟수: n(n-1)/2
    - 각 반복에서 1번의 비교가 수행됨
    - 각 단계에서 i-1번의 비교가 수행됨
  - 총 이동 횟수: n(n-1)/2
    - 각 단계에서 i번의 이동이 수행됨

## [3,7,9,4,1,6]때의 시뮬레이션

[InsertionSort-pinned Simulation](https://raw.githubusercontent.com/MisileLab/h3/main/projects/dsb/ans/simulations/InsertionSortPinned.mp4)

## 시뮬레이션

[InsertionSort Simulation](https://raw.githubusercontent.com/MisileLab/h3/main/projects/dsb/ans/simulations/InsertionSort.mp4)

# 버블 정렬

속도: O(n^2)

## 안정성이 있는 이유

자기 전과 자기의 요소만 바꾸기 때문에 같을 경우 교환이 일어나지 않고, 다를 경우 순서는 유지된 채로 교환이 되기 때문

## 초기 변수

- k는 배열 크기

## 방법

- n=1
- n과 n-1번째 원소를 비교하여 정렬하고, n을 1 증가시킴
- n이 k를 넘으면 종료
- 만약 전 이유로 종료했는데, 아무 교환도 하지 않았으면 정렬 완료
- k가 1일 경우 정렬 완료, 아닐 경우 k를 1 낮추고 첫번째부터 시작

## 성능

- 가장 빠르게 정렬되는 경우(이미 정렬된 경우)
  - 비교 횟수: n-1
    - 만약 교환 없이 비교만 될 경우 early-exit한다는 가정하에 n-1번만 이루어짐
  - 이동 횟수: 0
    - 이미 정렬되어 있기 때문에 이동하지 않음
- 가장 느리게 정렬되는 경우(역순 정렬된 경우)
  - 비교 횟수: n(n-1)/2
    - 각 요소를 비교해야 하지만, 한번 할 때마다 n-1번 비교하는 것이 아님
  - 이동 횟수: n(n-1)/2
    - 모든 비교 후 이동해야 하기 때문에 비교 횟수와 똑같음

## 시뮬레이션

[BubbleSort Simulation](https://raw.githubusercontent.com/MisileLab/h3/main/projects/dsb/ans/simulations/BubbleSort.mp4)
