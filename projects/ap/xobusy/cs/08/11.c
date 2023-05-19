#include <stdio.h>

int main() {
    int a, i;
    printf("구구단 단 입력 : ");
    scanf("%d", &a);
    for (i=1;i<=9;i++) {
        printf("%d * %d = %2d\n", a, i, a*i);
    }
}
