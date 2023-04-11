#include <stdio.h>

int main() {
    for (int i=1;i<=5;i++) {
        for (int j=4;j>=0;j--) {
            printf("%3d", j*5+i);
        }
        printf("\n");
    }
}
