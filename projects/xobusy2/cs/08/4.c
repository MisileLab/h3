#include <stdio.h>

int main() {
    int a, i;
    int sum = 0;
    printf("정수 입력 : ");
    scanf("%d", &a);
    for (i=1;i<=a;i++) {
        sum += i;
    }
    printf("1부터 %d까지의 합은 %d입니다.", a, sum);
}
