#include <stdio.h>

int main() {
    int sum = 0;
    int i;
    for (i=1;i<=100;i++) {
        if (i % 2 == 0) {
            sum -= i;
        } else {
            sum += i;
        }
    }
    printf("1-2+3-4+5-6+....+99-100=%d", sum);;
}
