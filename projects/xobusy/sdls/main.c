#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_render.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include "draw.h"
#include "input.h"
#include "image.h"
#include <stdbool.h>
#include <time.h>
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

SDL_Texture* makeText(SDL_Renderer* renderer, TTF_Font* font, const char* text, SDL_Color color, int x, int y) {
	SDL_Surface* surface = TTF_RenderText_Solid(font, text, color);
	SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
	SDL_Rect rect = { x, y, surface->w, surface->h };
	SDL_RenderCopy(renderer, texture, NULL, &rect);
	SDL_FreeSurface(surface);
	return texture;
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

	// Initialize SDL_ttf library
	if (TTF_Init() == -1) {
		printf("TTF_Init: %s\n", TTF_GetError());
		return 1;
	}

	// Load font file
	TTF_Font* font = TTF_OpenFont("a.ttf", 28);
	if (!font) {
		printf("TTF_OpenFont: %s\n", TTF_GetError());
		return 1;
	}

	// Set up text surface and texture
	SDL_Color textColor = { 255, 255, 255 };
	SDL_Surface* textSurface = TTF_RenderText_Solid(font, "Hello, I'm a chatbot!", textColor);
	SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

	// Get text dimensions
	int textWidth = textSurface->w;
	int textHeight = textSurface->h;

	// Set text position
	int textX = 20;
	int textY = SCREEN_HEIGHT - textHeight - 20;

	// Render text
	SDL_Rect textRect = { textX, textY, textWidth, textHeight };
	SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

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
	SDL_FreeSurface(textSurface);
	SDL_DestroyTexture(textTexture);
	TTF_CloseFont(font);
	TTF_Quit();
	SDL_Quit();

	return 0;
}
