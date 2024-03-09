#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#define MAX_SIZE 100
#define RAND_SIZE 30

void crack(int* entropy, int crack_speed);
void pol_calc(int* pols, char character, int streak);
void time_print(int time);
void pw_random();

int main() {
    int entropy = 0;
    int crack_speed = 100000;
    crack(&entropy, crack_speed);
    printf("Entropy: %d\n", entropy);
    time_print(entropy);
    bool a;
    srand(time(NULL));
    while (true) {
        printf("\nIf you want to change password if you think the password is easy to crack.\ndo you want it?(y/n)");
        char ans;
        scanf("%s", &ans);
        if (strcmp(&ans, "y") == 0) {
            a = true;
            break;
        } else if (strcmp(&ans, "n") == 0) {
            a = false;
            break;
        } else {
            printf("\nInvaild answer");
        }
    }
    if (a) {
        for (int i = 0; i<RAND_SIZE; i++) {
            pw_random();
        }
        printf("\n");
    }
}

void time_print(int time) {
    printf("Your password can cracked in: ");
    printf("%d seconds ", time % 60);
    if (time >= 60) {
        time /= 60;
        printf("%d minutes ", time % 60);
    } else return;
    if (time >= 60) {
        time /= 60;
        printf("%d days ", time % 60);
    } else return;
    if (time >= 24) {
        time /= 24;
        printf("%d days ", time % 60);
    } else return;
    if (time >= 365) {
        printf("%d years", time / 365);
    }
}

void crack(int* entropy, int crack_speed) {
    char password[MAX_SIZE];
    int pols[4] = {0,0,0,0}; // upper, lower, special, numbers
    int ents[4] = {26,30,50,10}; // upper, lower, special, numbers
    int streak = 0;

    printf("password: ");
    scanf("%s", password);
    int len_password = strlen(password);

    for (int i=0;i<len_password;i++) {
        if (i != 0 && password[i] == password[i-1]) {
            streak += 1;
        } else {
            streak = 0;
        }
        pol_calc(pols, password[i], streak);
    }

    for (int i=0;i<4;i++) {
        if (pols[i] > 0) {
            *entropy += pols[i] * ents[i];
        }
    }
}

void pol_calc(int* pols, char character, int streak) {
    if (character >= 'A' && character <= 'Z') {
        pols[0] += 4-streak;
    } else if (character >= 'a' && character <= 'z') {
        pols[1] += 4-streak;
    } else if (character >= '0' && character <= '9') {
        pols[3] += 4-streak;
    } else {
        pols[2] += 4-streak;
    }
}

void pw_random() {
    int random = rand() % 4;
    if (random == 0) {
        printf("%c", rand() % 26 + 'A');
    } else if (random == 1) {
        printf("%c", rand() % 26 + 'a');
    } else if (random == 2) {
        printf("%c", rand() % 10 + '0');
    } else {
        printf("%c", rand() % 33 + '!');
    }
}
