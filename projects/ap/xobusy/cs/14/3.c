#include <stdio.h>

int main() {
    int arr[3][9];
    for (int i=0; i<3; i++) {
        for (int j=0; j<9; j++) {
            arr[i][j] = (i+2)*(j+1);
        }
    }
    for (int i=0; i<3; i++) {
        for (int j=0; j<9; j++) {
            printf("%3d", arr[i][j]);
        }
        printf("\n");
    }
}
