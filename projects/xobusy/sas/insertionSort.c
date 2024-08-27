#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TYPE int
#define MAX_SIZE 10

int list[MAX_SIZE];
int n;

void swap(TYPE* x, TYPE* y) {printf("swap: %d %d", *x, *y);TYPE t=*x;*x=*y;*x=t;}

void insertionSort(int arr[], int n) {
  // 맨 앞은 정렬되어 있다고 가정함
  for (int i=1;i<n;i++) {
    int k = arr[i];
    // 자기 바로 뒤부터 검사
    int j = i-1;
    // j>=0 (인덱스가 -1이면 끝까지 검사한것)
    // arr[j] > k (arr[j]>arr[i]) (검사하는 것보다 크면 그 앞칸에 저장함)
    while (j>=0 && arr[j] > k) {
      arr[j+1] = arr[j];
      j--;
    }
    arr[j + 1] = k;
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
  insertionSort(list, n);
  printf("\nAfter Sorting:\n");
  for (i=0; i<n; i++) {printf("%d ", list[i]);}
  printf("\n");
}

