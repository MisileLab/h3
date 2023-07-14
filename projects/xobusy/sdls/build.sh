clang -O2 `sdl2-config --libs` -I draw.h draw.c -I input.h input.c -I image.h image.c main.c -o main.out
