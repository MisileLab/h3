#include <stdio.h>
#include <string.h>

int main() {
    char a[100000];
    int b=0, c=0;
    printf("문자열 입력 : ");
    scanf("%s", a);
    for (int i=0; a[i] != '\0' && i < 100000; i++) {
        if (a[i] == '(') {
            b++;
        } else if (a[i] == ')') {
            c++;
        }
    }
    printf("여는 괄호 : %d, 닫는 괄호 : %d", b, c);
}
