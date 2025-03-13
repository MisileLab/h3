#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TYPE int
#define MAX_SIZE 10

void swap(TYPE* x, TYPE* y) {
  TYPE t=*x;
  *x=*y;
  *y=t;
}

int list[MAX_SIZE];
int n;

int part(int arr[], int low, int high) {
  int pivot = arr[high];
  int i = low-1;
  for (int j=low; j<high; j++) {
    if (arr[j] < pivot) {
      i++;
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[i+1], &arr[high]);
  return i+1;
}

void quickSort(int list[], int low, int high) {
  if (low < high) {
    // 나눈 부분
    int pi = part(list, low, high);
    
    quickSort(list, low, pi-1);
    quickSort(list, pi+1, high);
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
  quickSort(list, 0, n-1);
  printf("\nAfter Sorting:\n");
  for (i=0; i<n; i++) {printf("%d ", list[i]);}
  printf("\n");
}

