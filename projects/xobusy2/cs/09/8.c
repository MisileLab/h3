#include <stdio.h>

int main() {
    int a, b;
    printf("행 갯수 입력 : ");
    scanf("%d", &a);
    printf("열 갯수 입력 : ");
    scanf("%d", &b);
    for (int i=1;i<=a;i++) {
        for (int i2=1;i2<=b;i2++) {
            printf("%c ", 0x40+i2);
        }
        printf("\n");
    }
}
