#include <stdio.h>

int main() {
    int status[2][4] = {{4, 6, 5, 8},{10, 96, 14, 108}};
    for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
            printf("%4d", status[i][j]);
        }
        printf("\n");
    }
    int arr[2][3] = {{1, 2}, {3, 4}};
}
