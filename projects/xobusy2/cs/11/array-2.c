#include <stdio.h>
int main() {
	int scores[5] = {88, 76, 93, 95, 68};
	int sum = 0;
	double avg;
	
	for(int i = 0; i< sizeof(scores)/sizeof(int); i++){
		sum += scores[i];
	}
	avg = (double)sum / (sizeof(scores)/sizeof(int));
	
	printf("%.1lf", avg);
}
