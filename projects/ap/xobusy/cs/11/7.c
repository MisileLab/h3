#include <stdio.h>
int main(){
	int va[5]={15, 20, 30};
	int vb[5]={0};

	printf("복사 전\n");
	printf("va : ");
	for(int i=0; i<5; i++){
		printf("%-3d ", va[i]);
	}
	printf("\nvb : ");
	for(int i=0; i<5; i++){
		printf("%-3d ", vb[i]);
	}
	
    for (int i=0;i<=sizeof va / sizeof(int);i++) {
        vb[i] = va[i];
    }
	
	printf("\n복사 후\n");
	printf("va : ");
	for(int i=0; i<5; i++){
		printf("%-3d ", va[i]);
	}
	printf("\nvb : ");
	for(int i=0; i<5; i++){
		printf("%-3d ", vb[i]);
	}
	
	
	return 0 ; 
}
