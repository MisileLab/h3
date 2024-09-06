#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define TYPE int
#define MAX_SIZE 10

int list[MAX_SIZE];
int n;

void swap(TYPE* x, TYPE* y) {TYPE t=*x;*x=*y;*y=t;}

void shellSort(int arr[], int n) {
  int h = n;
  while (h > 1) {
    h /= 2;
    if (h % 2 == 0) {h++;}
    for (int i = 0; i < h; i++) {
      for (int j = i + h; j < n; j += h) {
        int k = arr[j];
        int l = j - h;
        while (l >= i && arr[l] > k) {
          swap(&arr[l], &arr[l + h]);
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

