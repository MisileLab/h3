#include <stdio.h>

int main() {
    int a = -10;
    do {
        printf("정수 입력 : ");
        scanf("%d", &a);
    } while (a <= 0);
    printf("입력한 정수는 %d입니다.", a);
}
