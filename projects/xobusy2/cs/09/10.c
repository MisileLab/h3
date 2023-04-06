#include <stdio.h>

int main() {
    printf("\n");
    for (int i = 1; i <= 5; i++) {
        for (int i2 = i; i2 >= 1; i2--) {
            printf("*");
        }
        printf("\n");
    }
}
