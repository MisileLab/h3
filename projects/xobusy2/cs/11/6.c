#include <stdio.h>

int main() {
    int a[] = {33, 67, 23, 87, 95, 47, 75};
    int max = a[0];
    int min = a[0];
    for (int i=1;i<=6;i++) {
        if (a[i] > max) {
            max = a[i];
        }
        if (min > a[i]) {
            min = a[i];
        }
    }
    printf("max = %d, min = %d", max, min);
}
