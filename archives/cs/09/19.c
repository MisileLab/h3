#include <stdio.h>

int main() {
    printf("줄입력(홀수) : ");
    int a, i;
    scanf("%d", &a);
    for (i = 1; i <= a / 2 + 1; i++) {
        for (int j = 1; j <= i; j++) {
            printf("* ");
        }
        printf("\n");
    }
    for (int i = a / 2; i >= 1; i--) {
        for (int i2 = i; i2 >= 1; i2--) {
            printf("* ");
        }
        printf("\n");
    }
    printf("\n");
}
