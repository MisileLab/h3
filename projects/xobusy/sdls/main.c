#include <stdio.h>
#include <SDL2/SDL.h>
#include <stdlib.h>

#define VIDEO 0x01
#define AUDIO 0x02
#define KEY 0x04
#define MOUSE 0x08
#define JOYSTICK 0x10
#define SPEAKER 0x20
#define USB 0x40
#define ALL 0xFF

typedef struct node Node;
typedef struct list List;
struct node {
    Node* next;
    void* val;
};
struct list {
    Node* head;
    Node* tail;
    int len;
};
void list_init(List* list) {
    list->head = NULL;
    list->tail = NULL;
    list->len = 0;
};
void list_push(List* list, void* data) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->next = NULL;
    newNode->val = data;
    if (list->head == NULL) {
        list->head = newNode;
        list->tail = newNode;
    } else {
        list->tail->next = newNode;
        list->tail = newNode;
    }
    list->len++;
}
int list_insert(List* list, int index, void* data) {
    if (index < 0 || index > list->len) return 1;
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->val = data;
    if (index == 0) {
        newNode->next = list->head;
        list->head = newNode;
    }
    return 0;
}
void* list_pop(List* list) {
    if (list->len == 0) return NULL;
    Node* node = list->head;
    void* val = node->val;
    list->head = node->next;
    free(node);
    list->len--;
    return val;
}

void list_release(List* list) {
    Node* current = list->head;
    while (current != NULL) {
        Node* next = current->next;
        free(current->val);
        free(current);
        current = next;
    }
    list->head = NULL;
    list->tail = NULL;
    list->len = 0;
}

int initSDL() {
    if (SDL_Init(ALL) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }
    List list;
    list_init(&list);
    int a = 10;int b = 10;int c = 10;
    list_push(&list, &a);
    list_push(&list, &b);
    list_push(&list, &c);
    Node* cur = list.head;
    while (cur != NULL) {
        printf("%d\n", *(int*)cur->val);
        cur = cur->next;
    }
    return 0;
}

int main() {
    initSDL();
}
