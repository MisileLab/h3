#include <stdio.h>

int main() {
  int a[5];
  printf("배열에 입력된 값 : ");
  scanf("%d %d %d %d %d", &a[0], &a[1], &a[2], &a[3], &a[4]);
  double avg = 0.0;
  for (int i = 0; i < 5; i++) {
    avg += a[i];
  }
  printf("평균값 출력 : %.1f", avg / 5);
}
