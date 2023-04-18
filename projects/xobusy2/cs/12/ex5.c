#include <stdio.h>

int main() {
    int arr[10];
    int ampl;
    printf("정수 입력 : ");
    scanf("%d", &ampl);
    arr[0] = 0;
    for (int i=0;i<10;i++) {
        arr[i] = i*ampl;
    }
    for (int i=0;i<10;i++) {
        printf("%d ", arr[i]);
    }
}
