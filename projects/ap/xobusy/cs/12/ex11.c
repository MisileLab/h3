#include <stdio.h>

int main() {
    int a;
    printf("정수 입력 : ");
    scanf("%d", &a);
    int arr[] = {34, 12, 67, 22, 50};
    for (int i=0;i<5;i++) {
        if (arr[i] == a) {
            printf("검색 성공!! %d번 인덱스에 있습니다.", i);
            return 0;
        }
    }
    printf("검색 실패!!");
}
