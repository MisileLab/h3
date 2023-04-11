#include <stdio.h>

int main() {
    printf("줄입력(홀수) : ");
    int a;
    scanf("%d", &a);
    for (int i=a;i>=1;i-=2) {
        int j = (i / 2) * 2;
        for (int i2=i / 2;i2>=1;i2--) {
            printf(" ");
        }
        for (int i2=a-j;i2>=1;i2--) {
            printf("*");
        }
        for (int i2=i / 2;i2>=1;i2--) {
            printf(" ");
        }
        printf("\n");
    }
    for (int i=3;i<=a;i+=2) {
        int j = (i / 2) * 2;
        for (int i2=i / 2;i2>=1;i2--) {
            printf(" ");
        }
        for (int i2=a-j;i2>=1;i2--) {
            printf("*");
        }
        for (int i2=i / 2;i2>=1;i2--) {
            printf(" ");
        }
        printf("\n");
    }
}
