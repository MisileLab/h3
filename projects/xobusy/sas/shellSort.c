#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define TYPE int
#define MAX_SIZE 10

int list[MAX_SIZE];
int n;

void shellSort(int arr[], int n) {
  // h를 높이와 똑같이 설정
  int h = n;
  // h가 1이면 정렬이 완료된 것
  while (h > 1) {
    // h를 처음에 2로 나눔
    h /= 2;
    if (h % 2 == 0) {h++;}
    // h의 값과 부분 리스트 개수는 같기 때문에 for문을 사용함
    for (int i = 0; i < h; i++) {
      // h만큼 떨어진 리스트 생성
      for (int j = i + h; j < n; j += h) {
        // 부분 리스트를 삽입 정렬함
        int k = arr[j];
        int l = j - h;
        while (l >= i && arr[l] > k) {
          arr[l+h] = arr[l];
          l -= h;
        }
        arr[l + h] = k;
      }
    }
  }
}
int main() {
  int i;
  n = MAX_SIZE;
  srand(time(NULL));
  for (i=0; i<n; i++) {list[i] = rand() % 100;}
  printf("Before Sorting:\n");
  for (i=0; i<n; i++) {printf("%d ", list[i]);}
  printf("\ntry sorting:\n");
  shellSort(list, n);
  printf("\nAfter Sorting:\n");
  for (i=0; i<n; i++) {printf("%d ", list[i]);}
  printf("\n");
}

