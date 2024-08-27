#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TYPE int
#define MAX_SIZE 10

int list[MAX_SIZE];
int n;

void swap(TYPE* x, TYPE* y) {TYPE t=*x;*x=*y;*x=t;}

void bubbleSort(int arr[], int n) {
  for (int i=0; i<n; i++) {
    for (int j=0; j<i; j++) {
      printf("\n%d %d", arr[j], arr[j+1]);
      if (arr[j]<arr[j+1]) {printf("\n");for(int k=0;k<n;k++){printf("%d ", arr[k]);};swap(&arr[j], &arr[j+1]);}
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
  bubbleSort(list, n);
  printf("\nAfter Sorting:\n");
  for (i=0; i<n; i++) {printf("%d ", list[i]);}
  printf("\n");
}

