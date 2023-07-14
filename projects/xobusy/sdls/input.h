#pragma once

#define KEY_NONE 0
#define KEY_DOWN 1
#define KEY_PRESS 2
#define KEY_UP 3

void updateKeyState();
int getKeyState(int keyCode);
void updateMouseState();
int getButtonState(int buttonCode);
void getMousePos(int* posx, int* posy);
