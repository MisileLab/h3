#include <stdio.h>
#define size 3

int main(){
	double arr[size];
	double tmp;
	
	printf("arr 배열 : ");
	//배열에 입력받기
	for(int i = 0; i < size; i++){
		scanf("%lf", &arr[i]);
	}
	printf("변경 전\n");
	//배열 원소 출력
	for(int i = 0; i < size; i++){
		printf("arr[%d]: %.1lf ", i, arr[i]);
	}
	//배열을 역순으로 저장하는 코드
	for(int i = 0; i < size/2; i++){
		tmp = arr[i];
		arr[i] = arr[size-i-1];
		arr[size-i-1] = tmp;
	}

	printf("\n변경 후\n");
	//배열 원소 출력
	for(int i = 0; i < size; i++){
		printf("arr[%d]: %.1lf ", i, arr[i]);
	}
}
