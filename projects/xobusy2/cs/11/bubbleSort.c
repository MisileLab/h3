#include <stdio.h>
int main(){
	int a[5]={66,55,33,77,44};
	int temp;
	printf("정렬 전 : ");
	for(int i=0 ; i<5 ; i++) {
        printf("%d ", a[i]);
    }

    for (int i=0; i<5; i++) {
        for (int j=1; j<5-i; j++) {
            if (a[j-1] > a[j]) {
                temp = a[j];
                a[j] = a[j-1];
                a[j-1] = temp;
            }
        }
    }

	printf("\n정렬 후 : ");
	for(int i=0 ; i<5 ; i++) {
        printf("%d ",a[i]);
    }
	return 0 ; 
}
