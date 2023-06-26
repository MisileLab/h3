#include<stdio.h>
#include<string.h>
int main() {
	int i, j;
	char input[150]={'\0'};
	char cmp[4]={'\0'};
	scanf("%s", input);
	for(i=0;i<100;i++){
		cmp[0]=input[0+i*3];
		cmp[1]=input[1+i*3];
		cmp[2]=input[2+i*3];
		if(input[i]==0)
			break;
		if (strcmp(cmp, "UUU") == 0|| strcmp(cmp, "UUC") == 0) {
        printf("ㅍ");
    }
    else if (strcmp(cmp, "UUA") == 0|| strcmp(cmp, "UUG") == 0|| strcmp(cmp, "CUU") == 0|| strcmp(cmp, "CUC") == 0|| strcmp(cmp, "CUA") == 0) {
        printf("ㄱ");
    }
    else if (strcmp(cmp, "CUG") == 0) {
        printf("ㄴ");
    }
    else if (strcmp(cmp, "AUU") == 0|| strcmp(cmp, "AUC") == 0|| strcmp(cmp, "AUA") == 0) {
        printf("ㅂ");
    }
    else if (strcmp(cmp, "AUG") == 0) {
        printf("ㅛ");
    }
    else if (strcmp(cmp, "GUU") == 0|| strcmp(cmp, "GUC") == 0|| strcmp(cmp, "GUA") == 0|| strcmp(cmp, "GUG") == 0) {
        printf("ㅔ");
    }
    else if (strcmp(cmp, "UCU") == 0|| strcmp(cmp, "UCC") == 0|| strcmp(cmp, "UCA") == 0|| strcmp(cmp, "UCG") == 0) {
        printf("ㄹ");
    }
    else if (strcmp(cmp, "CCU") == 0|| strcmp(cmp, "CCC") == 0) {
        printf("ㅅ");
    }
    else if (strcmp(cmp, "CCA") == 0|| strcmp(cmp, "CCG") == 0) {
        printf("ㄷ");
    }
    else if (strcmp(cmp, "ACU") == 0|| strcmp(cmp, "ACC") == 0) {
        printf("ㅎ");
    }
    else if (strcmp(cmp, "ACA") == 0|| strcmp(cmp, "ACG") == 0) {
        printf("ㅒ");
    }
    else if (strcmp(cmp, "GCU") == 0|| strcmp(cmp, "GCC") == 0) {
        printf("ㅗ");
    }
    else if (strcmp(cmp, "GCA") == 0|| strcmp(cmp, "GCG") == 0) {
        printf("ㅐ");
    }
    else if (strcmp(cmp, "UAU") == 0|| strcmp(cmp, "UAC") == 0) {
        printf("ㅠ");
    }
    else if (strcmp(cmp, "UAA") == 0|| strcmp(cmp, "UAG") == 0) {
        printf("?");
    }
    else if (strcmp(cmp, "CAU") == 0|| strcmp(cmp, "CAC") == 0) {
        printf("ㅜ");
    }
    else if (strcmp(cmp, "CAA") == 0|| strcmp(cmp, "CAG") == 0) {
        printf("ㅡ");
    }
    else if (strcmp(cmp, "AAU") == 0|| strcmp(cmp, "AAC") == 0) {
        printf("ㅏ");
    }
    else if (strcmp(cmp, "AAA") == 0|| strcmp(cmp, "AAG") == 0) {
        printf("ㅣ");
    }
    else if (strcmp(cmp, "GAU") == 0|| strcmp(cmp, "GAC") == 0) {
        printf("ㅁ");
    }
    else if (strcmp(cmp, "GAA") == 0|| strcmp(cmp, "GAG") == 0) {
        printf("ㅈ");
    }
    else if (strcmp(cmp, "UGU") == 0|| strcmp(cmp, "UGC") == 0) {
        printf("ㅊ");
    }
    else if (strcmp(cmp, "UGA") == 0|| strcmp(cmp, "UGG") == 0) {
        printf("ㅋ");
    }
    else if (strcmp(cmp, "CGU") == 0|| strcmp(cmp, "CGC") == 0) {
        printf("ㅕ");
    }
    else if (strcmp(cmp, "CGA") == 0|| strcmp(cmp, "CGG") == 0) {
        printf("ㅖ");
    }
    else if (strcmp(cmp, "AGU") == 0|| strcmp(cmp, "AGC") == 0) {
        printf("ㅌ");
    }
    else if (strcmp(cmp, "AGA") == 0|| strcmp(cmp, "AGG") == 0) {
        printf("ㅇ");
    }
    else if (strcmp(cmp, "GGU") == 0|| strcmp(cmp, "GGC") == 0) {
        printf("ㅓ");
    }
    else if (strcmp(cmp, "GGA") == 0 || strcmp(cmp, "GGG") == 0) {
        printf("ㅑ");
    }
	}
}