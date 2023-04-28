#include <stdio.h>

int main() {
    int a[5];
    for (int i=0;i<5;i++) {
        printf("a[%d] = ", i);
        scanf("%d", &a[i]);
    }
    for (int i=4;i>=0;i--) {
        printf("%d ", a[i]);
    }
}
