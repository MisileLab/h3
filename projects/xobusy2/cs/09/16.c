#include <stdio.h>

int main() {
    printf("\n");
    for (int i=5;i>=1;i-=2) {
        int j = (i / 2) * 2;
        for (int i2=i / 2;i2>=1;i2--) {
            printf(" ");
        }
        for (int i2=5-j;i2>=1;i2--) {
            printf("*");
        }
        for (int i2=i / 2;i2>=1;i2--) {
            printf(" ");
        }
        printf("\n");
    }
}
