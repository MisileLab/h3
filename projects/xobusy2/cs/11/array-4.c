#include <stdio.h>

int main() {
	int arr[5] = { 6,4,3,9,5 }, i, temp;
	
	printf("변경 전 배열 : ");
	for (i = 0; i < 5; i++) {
		printf("%d ", arr[i]);
	}

	for (i=1; i<5; i++) {
        if (arr[i-1] > arr[i]) {
			temp = arr[i-1];
			arr[i-1] = arr[i];
			arr[i] = temp;
		}
    }

	printf("\n변경 후 배열 : ");
	for (i = 0; i < 5; i++) {
		printf("%d ", arr[i]);
	}

}