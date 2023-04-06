#include <stdio.h>

int main() {
    printf("\n");
    for (int i=0;i<3;i++) {
        for (int i2=1;i2<=i;i2++) {
            printf("  ");
        }
        printf("* * * * ");
        printf("\n");
    }
}
