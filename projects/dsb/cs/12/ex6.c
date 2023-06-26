#include <stdio.h>

struct Question {
    int answer;
    int point;
};

typedef struct Question question;

int main() {
	printf("====채점 프로그램====\n");
    question questions[7] = {
        {.answer = 4, .point = 15},
        {.answer = 3, .point = 10},
        {.answer = 2, .point = 20},
        {.answer = 5, .point = 5},
        {.answer = 1, .point = 15},
        {.answer = 2, .point = 15},
        {.answer = 5, .point = 20}
    };
	int point = 0, amount = 0;
	for(int i=0; i<7; i++){
		printf("문제 %d번 학생 답안 : ", i+1);
        int answer;
        scanf("%d", &answer);
        if (answer == questions[i].answer){
            amount++;
            point += questions[i].point;
            printf("맞았습니다!!\n");
        } else {
            printf("틀렸습니다!!\n");
        }
	}
	
	printf("학생의 점수는 %d점이며, 맞은 개수는 %d개입니다.", point, amount);
}
