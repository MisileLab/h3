#include <stdio.h>

int main() {
    int a, b = 0;
    int i;
    for (i=1;i<=1000;i++) {
        if (i%2==0) {
            a += i;
        } else {
            b += i;
        }
    }
    printf("홀수의 합 = %d 짝수의 합 = %d", b, a);
}
