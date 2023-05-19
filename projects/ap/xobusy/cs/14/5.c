#include <stdio.h>

int main() {
    int score[2][4] = {{70, 88, 83, 96}, {60, 92, 87, 56}};
    int score2[4][2];
    for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
            score2[j][i] = score[i][j];
        }
    }
    for (int i=0; i<2; i++) {
        for (int j=0; j<4; j++) {
            printf("%d ", score2[j][i]);
        }
        printf("\n");
    }
}
