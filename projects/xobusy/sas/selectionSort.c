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

void selectionSort(int list[], int n) {
  int i, j, least, temp;
  for (int i=0; i<n; i++) {
    int min = i;
    for (int j=i+1; j<n; j++) {
      if (list[min]>list[j]) {min = j;}
    }
    swap(&list[i], &list[min]);
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
  selectionSort(list, n);
  printf("\nAfter Sorting:\n");
  for (i=0; i<n; i++) {printf("%d ", list[i]);}
  printf("\n");
}

