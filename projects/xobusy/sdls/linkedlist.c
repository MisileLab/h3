#include "linkedlist.h"
#include <stdlib.h>
#include <memory.h>

// 정의
void list_init(List* list)
{
	list->head = NULL;
	list->tail = NULL;
	list->len = 0;
}

void list_push(List* list, void* data)
{
	Node* newNode = (Node*)malloc(sizeof(Node));
	newNode->next = NULL;
	newNode->val = data;

	// 리스트가 비어있는 경우(len == 0)
	if (list->head == NULL) {
		list->head = newNode;
		list->tail = newNode;
	}
	else {
		// 맨 뒤에 노드를 추가해라!
		list->tail->next = newNode;
		list->tail = newNode;
	}
	list->len++;
}

int list_insert(List* list, int index, void* data)
{
	if (index < 0 || index > list->len)
		return 1;	// 유효하지 않은 인덱스

	Node* newNode = (Node*)malloc(sizeof(Node));
	newNode->val = data;

	// 첫번째 항목 넣을 때
	if (index == 0) {
		newNode->next = list->head;
		list->head = newNode;

		if (list->tail == NULL)
			list->tail = newNode;
	}
	// 중간이나 맨 뒤에 삽입할 때..
	else {
		Node* prevNode = list->head;
		// index 이전까지 이동
		for (int i = 1; i < index; i++) {
			prevNode = prevNode->next;
		}

		newNode->next = prevNode->next;
		prevNode->next = newNode;

		if (newNode->next == NULL)
			list->tail = newNode;
	}
	list->len++;
	return 0;
}

void* list_pop(List* list, int index)
{
	if (index < 0 || index > list->len)
		return NULL;	// 유효하지 않은 인덱스
	// 지울 노드 선언
	Node* removedNode;
	// 리스트의 맨 앞 노드 삭제
	if (index == 0) {
		removedNode = list->head;
		list->head = removedNode->next;

		if (list->tail == removedNode)
			list->tail = NULL;
	}
	// 리스트의 중간이나 맨 뒤 노드 삭제
	else {
		// 이전 노드를 찾는다.
		Node* prevNode = list->head;
		// index 이전까지 이동
		for (int i = 1; i < index; i++) {
			prevNode = prevNode->next;
		}
		removedNode = prevNode->next;
		prevNode->next = removedNode->next;
		if (removedNode == list->tail)
			list->tail = prevNode;
	}
	void* val = removedNode->val;
	free(removedNode);
	list->len--;
	return val;
}

void list_release(List* list)
{
	Node* currentNode = list->head;
	while (currentNode != NULL) {
		// 다음 노드 보관
		Node* nextNode = currentNode->next; 
		free(currentNode);	// 현재 지우고
		currentNode = nextNode;	// 현재를 다음 노드로 변경
	}
	list->head = NULL;
	list->tail = NULL;
	list->len = 0;
}

void *list_remove(List *list, void* data) {
	Node* prevNode = NULL;
	Node* currentNode = list->head;

	while (currentNode != NULL) {
		if (currentNode->val == data) {
			if (prevNode == NULL) {
				list->head = currentNode->next;
				if (list->tail == currentNode) {
					list->tail = NULL;
				}
			} else {
				prevNode->next = currentNode->next;
				if (list->tail == currentNode) {
					list->tail = prevNode;
				}
			}
			free(currentNode);
			list->len--;
			return data;
		}
	}

	return NULL;
}

void *list_search(List *list, void *data) {
	Node *currentNode = list->head;
	while (currentNode != NULL) {
		if (currentNode->val == data) {
			return currentNode->val;
		}
		currentNode = currentNode->next;
	}
	return NULL;
}

void list_releaseWVal(List* list)
{
	Node* currentNode = list->head;
	while (currentNode != NULL) {
		// 다음 노드 보관
		Node* nextNode = currentNode->next; 
		free(currentNode->val); // 노드에 할당된 값 메모리 해제
		free(currentNode);  // 노드 메모리 해제
		currentNode = nextNode; // 현재를 다음 노드로 변경
	}
	list->head = NULL;
	list->tail = NULL;
	list->len = 0;
}

int main() {
	
}
