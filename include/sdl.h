#ifndef __SDLSCREEN_H__
#define __SDLSCREEN_H__

#include <iostream>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "timer.h"





class sSDL{

	public:
	size_t nx, ny;
	float Lx, Ly;
	unsigned int pts;
	

	bool initSDL;
	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Texture* texture;
	SDL_Surface* screen_surface;
	SDL_Surface* text_surface;
	bool plotlegend;
	SDL_Color bgcolor;
	unsigned int bgcoloruint;
	SDL_Color color;
	


	unsigned int *cmap_rgba;
	unsigned int *cmap_rgba_data;
	unsigned int* pixels;
	unsigned int *plot_rgba_data;
	int pitch;
	int pitch1;


	uint screen_dimx, screen_dimy;

	unsigned int texXdim;
	unsigned int texYdim;
	int fontsize;
	TTF_Font* font;
	TTF_Font* font_title;
	std::string font_name;
	std::string font_title_name;
	int text_legend_w;
	int text_legend_h;
	int text_axis_w;
	int text_axis_h;
	int text_title_h;



	int barx, bardim, posx, posy;


	int ncol;

	float factor;


//	const Uint8 *keys;
	bool cexit;
	SDL_Event e;

	bool gpu;


	unsigned int stepsUpScreen;

	bool paused;

	void checkEvent(double &elapsedtime, Timer &grosstime, double time);
	void eventHandler(bool print, double &elapsedtime, Timer &grosstime, double time);
	//Interaction and visualization
	bool exitRequested() const;

	void finishSDL();

	void save_image(std::string filename);


	void updateScreen(float minvar0, float maxvar0, float minvar1, float maxvar1, float integral, float t, float t0);

	sSDL(){};
	~sSDL() {	finishSDL();	}

	void InitSDL(float factor_, size_t nx_, size_t ny_, float Lx_, float Ly_, bool gpu_);
};




SDL_Surface* My_TTF_Render(TTF_Font *font, const char *text, SDL_Color textColor);


void Render_Text(SDL_Surface *screen, TTF_Font *font, SDL_Color textColor, const char *text, int x, int y );

#endif
