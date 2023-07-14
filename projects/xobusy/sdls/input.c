#include "input.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_mouse.h>
#include <memory.h>
#include <stdlib.h>

size_t memsize = sizeof(Uint8) * SDL_NUM_SCANCODES;
Uint8 *currentKeyState = NULL;
Uint8 beforeKeyState[SDL_NUM_SCANCODES];
int currentMousePosX = 0;
int currentMousePoxY = 0;
Uint32 beforeMouseState = 0;
Uint32 currentMouseState = 0;

void updateKeyState() {
    if (currentKeyState == NULL) {
        currentKeyState = SDL_GetKeyboardState(NULL);
    }
    SDL_memcpy(beforeKeyState, currentKeyState, memsize);
}

int getKeyState(int keyCode) {
    if (!beforeKeyState[keyCode]) {
        if (currentKeyState[keyCode]) {
            return KEY_DOWN;
        } else {
            return KEY_NONE;
        }
    } else {
        if (currentKeyState[keyCode]) {
            return KEY_PRESS;
        } else {
            return KEY_UP;
        }
    }
}

void updateMouseState() {
    beforeMouseState = currentMouseState;
    currentMouseState = SDL_GetMouseState(&currentMousePosX, &currentMousePoxY);
}

int getButtonState(int buttonCode) {
    Uint32 buttonMask = SDL_BUTTON(buttonCode);
    Uint32 before = beforeMouseState & buttonMask;
    Uint32 current = currentMouseState & buttonMask;
    if (before) {
        if (current) {
            return KEY_PRESS;
        } else {
            return KEY_UP;
        }
    } else {
        if (current) {
            return KEY_NONE;
        } else {
            return KEY_DOWN;
        }
    }
}

void getMousePos(int *posx, int *posy) {
    *posx = currentMousePosX;
    *posy = currentMousePoxY;
}
