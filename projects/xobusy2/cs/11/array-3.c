#include <stdio.h>

int main() {
	int arr[5] = {6,4,3,9,5}, i, temp;
	int min=arr[0], minIndex=0;
	
	printf("변경 전 배열 : ");
	for(i=0; i<5; i++){
		printf("%d ", arr[i]);
	}
    
    for (i=0; i<5; i++) {
        int value = arr[i];
        if (value < min) {
            min = arr[i];
            minIndex = i;
        }
    }
    int tmp = arr[0];
    arr[0] = arr[minIndex];
    arr[minIndex] = tmp;
	
	printf("\n변경 후 배열 : ");
	for(i=0; i<5; i++){
		printf("%d ", arr[i]);
	}
}
