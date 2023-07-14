#pragma once
// 선언

// 구조체 선언
typedef struct node Node;
typedef struct list List;

struct node {
	Node* next;		// 다음 노드
	void* val;		// 노드의 값
};

struct list {
	Node* head;		// 첫번째 노드
	Node* tail;		// 마지막 노드
	int len;		// 리스트의 길이
};

// 리스트를 초기화 합니다.
void list_init(List* list);
// 리스트 맨 뒤에 데이터를 추가합니다.
void list_push(List* list, void* data);
// 원하는 위치에 데이터를 추가합니다.
int list_insert(List* list, int index, void* data);
// 원하는 위치의 노드를 제거합니다.
void* list_pop(List* list, int index);
// 리스트 전체를 해제하는 함수
void list_release(List* list);
// 리스트 전체를 해제하는 함수
// void*도 동적 해제합니다.
void list_releaseWVal(List* list);
void *list_search(List* list, void* data);
void *list_remove(List* list, void* data);