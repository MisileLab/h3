#include <stdio.h>

int factorial(int b) {
  int i;
  int sum = 1;
  for (i=b;i>=1;i--) {
    sum *= i;
  }
  return sum;
}

int main() {
  int a;
  printf("정수 입력 : ");
  scanf("%d", &a);
  printf("%d! = %d", a, factorial(a));
}
