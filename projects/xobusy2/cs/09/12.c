#include <stdio.h>

int main() {
    printf("\n");
    for (int i = 5; i >= 1; i--) {
        for (int i2 = i; i2 >= 2; i2--) {
            printf(" ");
        }
        for (int i2 = (5-i); i2 >= 0; i2--) {
            printf("*");
        }
        printf("\n");
    }
}
