#include <stdio.h>

int main() {
    printf("몇개의 수를 입력하시겠습니까? : ");
    int a;
    scanf("%d", &a);
    int b[a];
    int tmp;
    printf("배열의 값 입력 : ");
    for (int j=0; j<a; j++) {
        scanf("%d", &b[j]);
    }
    for (int i=0; i<a; i++) {
        for (int j=0; j<a; j++) {
            printf("%d ", b[j]);
        }
        for (int j=0; j<a-1; j++) {
            tmp = b[j+1];
            b[j+1] = b[j];
            b[j] = tmp;
        }
        printf("\n");
    }
}
