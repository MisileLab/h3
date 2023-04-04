#include <stdio.h>

int main() {
	int i = 1;
    int sum = 0;
	int a;
    printf("정수입력 : ");
	scanf("%d", &a);
    while (i <= a) {
        sum += i;
        i++;
    }
    printf("%d까지의 합 = %d", a, sum);
}
