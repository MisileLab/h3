#include <stdio.h>

int main() {
    printf("\n");
    for (int i=1;i<=5;i++) {
        for (int j=0;j<i;j++) {
            printf("%c", 0x41+j);
        }
        printf("\n");
    }
}
