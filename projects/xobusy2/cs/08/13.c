#include <stdio.h>

int main() {
    int a, i;
    printf("별의 갯수 입력 : ");
    scanf("%d", &a);
    for (i=1;i<=a;i++) {
        printf("*");
        if (i%5 == 0) {
            printf("\n");
        }
    }
}
