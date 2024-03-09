#include <stdio.h>

int main() {
    printf("\n");
    for (int i=3;i>=1;i--) {
        for (int i2=2;i2<=i;i2++) {
            printf("  ");
        }
        printf("* * * * ");
        printf("\n");
    }
}
