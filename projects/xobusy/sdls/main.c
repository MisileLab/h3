#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_render.h>
#include <stdio.h>
#include "draw.h"
#include "input.h"
#include "image.h"
#include <stdbool.h>
#include <time.h>
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

void _repeat(SDL_Renderer* renderer, int x, int y) {
	SDL_SetRenderDrawColor(renderer, 20, 80, 172, 0xFF);
	SDL_RenderClear(renderer);
	// drawFilledCircle(renderer, 320 - 100, 240 - 40, 50, 0, 0, 0, 255);
	// drawFilledCircle(renderer, 320 + 100, 240 - 40, 50, 0, 0, 0, 255);
	// drawFilledTriangle(renderer, 320, 240, 320 - 20, 240 + 50, 320 + 20, 240 + 50, 255, 0, 255, 255);
	drawFilledRectangle(renderer, x, y, 100, 200, 0, 0, 0, 255);
	SDL_RenderPresent(renderer);
}

int main() {
	SDL_Window* window = NULL;
	SDL_Surface* renderer = NULL;

	if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		printf("SDL moment. %s\n", SDL_GetError());
	}
	if (!initImage()) {
		printf("no u");
		SDL_Quit();
		return 1;
	}

	window = SDL_CreateWindow("SDL Project",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		SCREEN_WIDTH, SCREEN_HEIGHT, 0);
	if (!window) {
		printf("no window: %s \n", SDL_GetError());
		SDL_Quit();
		return 1;
	}
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	if (!renderer) {
		printf("no renderer: %s \n", SDL_GetError());
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 0xFF);
	SDL_RenderClear(renderer);
	//SDL_Texture* texture = loadTexture(renderer, "archbtw.png");
	//drawTexture(renderer, 0, 0, texture);
	drawFilledRectangle(renderer, 320, 100, 10, 100, 255, 0, 0, 255);
	SDL_RenderPresent(renderer);
	drawFilledRectangle(renderer, 320, 200, 100, 10, 255, 0, 0, 255);
	SDL_RenderPresent(renderer);
	drawFilledCircle(renderer, 330, 200, 50, 255, 0, 0, 255);
	SDL_RenderPresent(renderer);
	drawFilledTriangle(renderer, 320 + 20, 240, 320, 240 + 50, 320 - 20, 240 + 50, 255, 0, 0, 255);
	SDL_RenderPresent(renderer);
	drawFilledCircle(renderer, 250, 150, 50, 255, 255, 0, 255);
	SDL_RenderPresent(renderer);

	SDL_Event event;
	struct timespec ts = {0, 100};

	bool quit = false;
	int x = 320; int y = 240;
	while (!quit) {
		updateKeyState();
		updateMouseState();
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				quit = true;
				break;
			}
		}
	}
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}
