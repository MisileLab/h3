#include <stdio.h>

int main() {
    int i;
    for (i=1;i<=50;i++) {
        printf("%6d", i);
        if (i % 5 == 0) {
            printf("\n");
        }
    }
}
