#include <stdio.h>

int main() {
    int a;
    int sum = 0;
    for (a=1;a<=1000;a++) {
        sum += a;
    }
    printf("1부터 1000까지의 합은 %d입니다.", sum);
}
