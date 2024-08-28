#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define TYPE int
#define MAX_SIZE 10

int list[MAX_SIZE];
int n;

void swap(TYPE* x, TYPE* y) {TYPE t=*x;*x=*y;*y=t;}

void bubbleSort(int arr[], int n) {
  // 맨 뒤는 정렬되어있다고 가정
  for (int i=0; i<n-1; i++) {
    bool swapped = false;
    // 자기보다 뒤는 정렬되어 있다고 가정
    for (int j=0; j<n-i-1; j++) {
      if (arr[j]>arr[j+1]) {
        swap(&arr[j], &arr[j+1]);
        swapped = true;
      }
    }

    // 만약 아무것도 안 바뀌면 모두 정렬된거임
    if (swapped == false) {break;}
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
  bubbleSort(list, n);
  printf("\nAfter Sorting:\n");
  for (i=0; i<n; i++) {printf("%d ", list[i]);}
  printf("\n");
}

