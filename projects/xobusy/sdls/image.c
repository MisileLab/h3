#include <SDL2/SDL_image.h>
#include <SDL2/SDL.h>
#include "image.h"
#include <stdio.h>

int initImage() {
    int imgFlags = IMG_INIT_PNG | IMG_INIT_JPG;
    if (!(IMG_Init(imgFlags) & imgFlags)) {
        printf("Failed to initialize SDL_image: %s\n", IMG_GetError());
        return 0;
    }
    return 1;
}

SDL_Texture* loadTexture(SDL_Surface* renderer, const char* file) {
    SDL_Surface* surface = IMG_Load(file);
    SDL_Texture* texture;
    if (surface == NULL) {
        printf("error bro");
        return NULL;
    }

    texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (texture == NULL) {
        printf("no texture");
        return NULL;
    }
    SDL_FreeSurface(surface);
    return texture;
}

void drawTexture(SDL_Surface *renderer, int x, int y, SDL_Texture *texture) {
    SDL_Rect src;
    SDL_Rect dst;

    src.x = 0;
    src.y = 0;
    SDL_QueryTexture(texture, NULL, NULL, &src.w, &src.h);
    dst.x = x;
    dst.y = y;
    dst.w = src.w;
    dst.h = src.h;
    SDL_RenderCopy(renderer, texture, NULL, &dst);
}
