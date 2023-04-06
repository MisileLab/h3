#include <stdio.h>

int main() {
    int sum = 1;
    int i;
    for (i=1;i<=19;i++) {
        if ((i-1)%3==0) {
            sum *= i;
        }
    }
    printf("1*4*7*10*13*16*19=%d", sum);
}
