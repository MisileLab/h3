#include <stdio.h>
int main() {
    int snack, students;
    scanf("%d", &students);
    scanf("%d", &snack);
    if (snack < students) {
        printf("0");
    } else if (snack == students) {
        printf("%d", students);
    } else {
        while (snack > 0) {
            snack -= students;
        }
        printf("과자를 다 주었습니다");
        // 정확히 기억 안남
    }
}
