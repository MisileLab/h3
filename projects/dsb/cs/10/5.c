#include <stdio.h>

int main() {
    for (int i=5;i>=1;i--) {
        for (int j=0;j<=4;j++) {
            printf("%3d", j*5+i);
        }
        printf("\n");
    }
}
