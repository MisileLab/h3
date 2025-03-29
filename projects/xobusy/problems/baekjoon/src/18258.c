#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#define uint uint32_t

struct Node {
  uint value;
  struct Node* next;
};

struct Queue {
  struct Node* head;
  struct Node* tail;
  uint size;
};

void push(struct Queue* queue, uint value) {
  queue->size++;
  struct Node* node = (struct Node*)malloc(sizeof(struct Node));
  node->value = value;
  node->next = NULL;
  if (queue->tail != NULL) {
    queue->tail->next = node;
  }
  queue->tail = node;
  if (queue->head == NULL) {
    queue->head = node;
  }
}

uint pop(struct Queue* queue) {
  queue->size--;
  if (queue->head == NULL) {
    return UINT32_MAX;
  }
  uint v = queue->head->value;
  struct Node* head = queue->head;
  queue->head = head->next;
  free(head);
  return v;
}

bool cmp(char* v, char* v2) {
  return strcmp(v, v2) == 0;
}

int main() {
  struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
  queue->size = 0;
  queue->head = NULL;
  queue->tail = NULL;
  int number;
  scanf("%d", &number);
  char line[6];
  for (int i = 0; i < number; i++) {
    scanf("%s", line);
    if (cmp(line, "back")) {
      if (queue->tail == NULL) {
        printf("-1\n");
      } else {
        printf("%d\n", queue->tail->value);
      }
    } else if (cmp(line, "front")) {
      if (queue->head == NULL) {
        printf("-1\n");
      } else {
        printf("%d\n", queue->head->value);
      }
    } else if (cmp(line, "empty")) {
      printf("%d\n", queue->size == 0);
    } else if (cmp(line, "size")) {
      printf("%d\n", queue->size);
    } else if (cmp(line, "pop")) {
      if (queue->size == 0) {
        printf("-1\n");
      } else {
        uint v = pop(queue);
        if (v == UINT32_MAX) {
          printf("-1\n");
        } else {
          printf("%d\n", v);
        }
      }
    } else {
      uint value;
      scanf("%u", &value);
      push(queue, value);
    }
  }
}
