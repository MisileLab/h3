#include <stdio.h>
int main(){
   int a[ ] ={3, 2, 7, 6, 9}, i;
   int b[5] ={3, 2, 7};  // Q) 배열 b의 원소는?
   int c[5] ={3, 2, 7,}; // Q) 배열 c의 원소는?
	
   printf("a : ");
   for(i=0 ; i<5 ; i++){
       printf("%3d", a[i]);
   }
   printf("\nb : ");
    for(i=0 ; i<5 ; i++){
       printf("%3d", b[i]);
   }
   printf("\nc : ");
   for(i=0 ; i<5 ; i++){
       printf("%3d", c[i]);
   }
   return 0 ; 
}
