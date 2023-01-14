#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include "libs/misilelib-c/misilelib.c"
#define MAX_LEN 100

void cat() {
    char filename[MAX_LEN];
    printf("파일 이름: ");
    scanf("%s", filename);
	printf("\n");
    FILE* fs;
    fs = fopen(filename, "r");
    while (feof(fs) == 0) {
        char str[MAX_LEN];
        fgets(str, MAX_LEN, fs);
        printf("%s", str);
    }
    fclose(fs);
}

void fls() {
    DIR *d;
    struct dirent *dir;
    d = opendir(".");
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            printf("%s\n", dir->d_name);
        }
        closedir(d);
    }
}

void cd() {
    char dirname[MAX_LEN];
    printf("디렉토리 이름: ");
    scanf("%s", dirname);
    chdir(dirname);
}

int main() {
    bool a = true;
    char* commands[] = {
        "cat",
        "exit",
        "ls",
        "cd",
		"clear",
		"cdir"
    };

    while (a) {
        char buffer[MAX_LEN];
        printf("> ");
        scanf("%s", buffer);
        if (compare_strs(buffer, commands[0])) {
            cat();
        } else if (compare_strs(buffer, commands[1])) {
            return 0;
        } else if (compare_strs(buffer, commands[2])) {
            fls();
        } else if (compare_strs(buffer, commands[3])) {
            cd();
        } else if (compare_strs(buffer, commands[4])) {
            printf("\e[1;1H\e[2J");
        } else if (compare_strs(buffer, commands[5])) {
		    char cwd[MAX_LEN];
			if (getcwd(cwd, sizeof(cwd)) != NULL) {
				printf("현재 디렉토리: %s\n", cwd);
			} else {
				printf("error\n");
			}
		} else {
            printf("존재하지 않는 명령어입니다.\n");
        }
    }
}
