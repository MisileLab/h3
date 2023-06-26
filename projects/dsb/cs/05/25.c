#include <stdio.h>

int main() {
    int a;
    int i = 1;
    int sum = 0;
    while (sum < 100) {
        sum += i;
        i++;
    }
    printf("n = %d sum = %d", i-1, sum);
}
