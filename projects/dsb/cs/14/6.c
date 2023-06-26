#include <stdio.h>

int main() {
    int arr[4][4];
    int sum2 = 0;
    for (int i=0; i<4; i++) {
        scanf("%d %d %d %d", &arr[i][0], &arr[i][1], &arr[i][2], &arr[i][3]);
    }
    for (int i=0; i<4; i++) {
        int sum = 0;
        for (int j=0; j<4; j++) {
            sum += arr[i][j];
            printf("%5d", arr[i][j]);
        }
        printf("%5d\n", sum);
    }
    for (int i=0; i<4; i++) {
        int sum = 0;
        for (int j=0; j<4; j++) {
            sum += arr[j][i];
        }
        printf("%5d", sum);
        sum2 += sum;
    }
    printf("%5d\n", sum2);
}
