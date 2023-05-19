#include <stdio.h>

int fibonacci(int num) {
    if (num == 1 || num == 0) {
        return 1;
    } else {
        return fibonacci(num-1) + fibonacci(num-2);
    }
}

int main() {
    int fibo[10] = {0, 1};
    for (int i=2; i<=9; i++) {
        fibo[i] = fibonacci(i-1);
    }
    for (int i=0; i<=9; i++) {
        printf("%d ", fibo[i]);
    }
}
