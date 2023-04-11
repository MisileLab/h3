#include <stdio.h>
int main(){
  int i;
	//3, 2, 7, 6, 9를 원소로 하는 정수형 배열 a 선언하기
  int a[] = {3, 2, 7, 6, 9};

  printf("배열 a의 기억공간의 크기 : %lubyte\n", sizeof a);
  printf("배열 a의 원소 개수 : %lu개\n", sizeof a / sizeof(int));

	//배열 출력 방법 1: 원소 하나씩 출력
	printf("%-3d%-3d%-3d%-3d%-3d\n", a[0], a[1], a[2], a[3], a[4]);

	//배열 출력 방법 2 : for문으로 출력
  for(int i=0;i <= 4;i++) {
    printf("%-3d", a[i]);
  }

  printf("\n");

	// 배열의 원소를 거꾸로 출력
  for(int i=4;i >= 0;i--){
    printf("%-3d", a[i]);
  }

  return 0;
}
