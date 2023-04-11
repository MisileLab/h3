#include <stdio.h>

int main() {
    int a[5];
    for (int i=0;i < 5;i++) {
        printf("a[%d] = ", i);
        scanf("%d", &a[i]);
    }
    for (int i=0; i < 5;i++) {
        printf("a[%d]= %d ", i, a[i]);
    }
}
