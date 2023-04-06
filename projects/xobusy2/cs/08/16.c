#include <stdio.h>

int main() {
    printf("***두 수 구간의 숫자 출력하기***\n");
    printf("두 수를 입력하세요 : ");
    int a, b, min, max;
    scanf("%d %d", &a, &b);
    if (a > b) {
        max = a;
        min = b;
    } else {
        max = b;
        min = a;
    }
    for (int i=min;i<=max;i++) {
        printf("%d ", i);
    }
}
