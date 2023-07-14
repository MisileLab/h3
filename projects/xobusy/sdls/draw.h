#pragma once
#include <SDL2/SDL.h>

void drawFilledRectangle(SDL_Surface *renderer, int x, int y, int w, int h, Uint8 r, Uint8 g, Uint8 b, Uint8 a);
void drawFilledTriangle(SDL_Surface *renderer, int x1, int y1, int x2, int y2, int x3, int y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a);
void drawFilledCircle(SDL_Surface *renderer, int x, int y, int radius, Uint8 r, Uint8 g, Uint8 b, Uint8 a);
