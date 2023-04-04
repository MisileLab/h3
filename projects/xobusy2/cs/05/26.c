#include <stdio.h>
#include <stdbool.h>

int main() {
    int a = -1;
    while (true) {
        printf("정수 1개 입력 : ");
        scanf("%d", &a);
        if (a > 0) {
            break;
        }
        printf("양의 정수를 입력하세요!!\n");
    }
}
