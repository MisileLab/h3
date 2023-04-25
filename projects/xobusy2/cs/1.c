#include <stdio.h>

int main() {
    char str1[] = "sunrin";
    char str2[] = {1, 2, 3};
    printf("%d\n", str1);
    printf("%d\n", &str1[0]);
    printf("%x\n", str2);
    printf("%x\n", &str2[0]);
    char str[3] = {'s', 'u', 'n'};
    printf("%s", str);
}
