#include <stdio.h>
int main() {
	int binary[4] = {1,1,0,1};
    int decimal = binary[0];
	
    for (int i=1; i<4; i++) {
        printf("%d\n", decimal);
        decimal = (decimal << 1) + binary[i];
    }
	
	printf("%d", decimal);
}
