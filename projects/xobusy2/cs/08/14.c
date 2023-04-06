#include <stdio.h>

int main() {
    int a[5];
    int sum = 0;
    for (int i=1;i<=5;i++) {
        printf("나이를 입력하세요 : ");
        scanf("%d", &a[i]);
    }
    for (int i=1;i<=5;i++) {
        sum += a[i];
    }
    printf("가족의 평균 나이는 %.2f입니다.", (double)sum / 5);
}
