#include <stdio.h>

int main() {
    int arr[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    int arr2[4][2];
    for (int i=0; i<8; i++) {
        arr2[i/2][i%2] = arr[i/4][i%4];
    }
    for (int i=0; i<4; i++) {
        for (int j=0; j<2; j++) {
            printf("%d ", arr2[i][j]);
        }
        printf("\n");
    }
}
