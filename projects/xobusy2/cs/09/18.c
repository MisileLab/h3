#include <stdio.h>

int main() {
    int a;
    printf("줄 수 입력 : ");
    scanf("%d", &a);
    for (int i=1;i<=a;i++) {
        for (int j=0;j<i;j++) {
            printf("%c", 65+j);
        }
        printf("\n");
    }
}
