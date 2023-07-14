#pragma once
#include <SDL2/SDL_image.h>

int initImage();
SDL_Texture* loadTexture(SDL_Surface* renderer, const char* file);
void drawTexture(SDL_Surface* renderer, int x, int y, SDL_Texture* texture);
