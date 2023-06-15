#include <stdio.h>
#include <string.h>
#define MAX_SIZE 100

void crack(int* entropy, int crack_speed);
void pol_calc(int* pols, char character);
void time_print(int time);

int main() {
    int entropy = 0;
    int crack_speed = 10000000;
    crack(&entropy, crack_speed);
    time_print(entropy);
}

void time_print(int time) {
    printf("Your password can cracked in: ");
    printf("%d seconds ", time % 60);
    if (time >= 60) {
        time /= 60;
    }
    if (time >= 60) {
        printf("%d minutes ", time % 60);
        time /= 60;
    }
    if (time >= 24) {
        printf("%d days ", time % 60);
        time /= 24;
    }
    if (time >= 365) {
        printf("%d years", time / 365);
    }
}

void crack(int* entropy, int crack_speed) {
    char password[MAX_SIZE];
    int pols[4] = {0,0,0,0}; // upper, lower, special, numbers
    int ents[4] = {26,26,33,10}; // upper, lower, special, numbers

    printf("password: ");
    scanf("%s", password);
    int len_password = strlen(password);

    for (int i=0;i<len_password;i++) {
        pol_calc(pols, password[i]);
    }

    for (int i=0;i<len_password;i++) {
        if (pols[i] > 0) {
            *entropy += pols[i] * ents[i];
        }
    }
}

void pol_calc(int* pols, char character) {
    if (character >= 'A' && character <= 'Z') {
        pols[0] += 1;
    } else if (character >= 'a' && character <= 'z') {
        pols[1] += 1;
    } else if (character >= '0' && character <= '9') {
        pols[3] += 1;
    } else {
        pols[2] += 1;
    }
}
