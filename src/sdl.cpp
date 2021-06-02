
#include <iostream>

#include <fstream>

#include <string>
#include <stdexcept>
#include <cstdio>



#include "sdl.h"


#include "savepng.h"



#include <cuda.h>
#include <cuda_runtime_api.h>

#include "timer.h"
#include "constants.h"





SDL_Surface* My_TTF_Render(TTF_Font *font, const char *text, SDL_Color textColor){
	//return TTF_RenderText_Blended(font, text, textColor);
	return TTF_RenderText_Solid(font, text, textColor);
}	



void Render_Text(SDL_Surface *screen, TTF_Font *font, SDL_Color textColor, const char *text, int x, int y ) {
    SDL_Surface *text_surface = My_TTF_Render(font, text, textColor);
	if(!text_surface){
		printf("TTF_RenderText_Solid: %s\n", TTF_GetError());
		exit(-5);
	}
	SDL_Rect dst = {x, y, text_surface->w, text_surface->h};
	if (SDL_BlitSurface(text_surface, NULL, screen, NULL) < 0) {
		printf("SDL_BlitSurface Error: %s", SDL_GetError());
		exit(1);
	}
	SDL_FreeSurface(text_surface);
}





void RendererText(SDL_Renderer* renderer, TTF_Font *font, SDL_Color textColor, const char *text, int x, int y ) {
    SDL_Surface *text_surface = My_TTF_Render(font, text, textColor);
	if(!text_surface){
		printf("TTF_RenderText: %s\n", TTF_GetError());
		exit(-5);
	}
	SDL_Texture* sTexture = SDL_CreateTextureFromSurface( renderer, text_surface );
	SDL_FreeSurface(text_surface);

    SDL_Rect Rect;
	SDL_QueryTexture( sTexture, NULL, NULL, &Rect.w, &Rect.h );
	Rect.x = x;
	Rect.y = y;
	SDL_RenderCopy( renderer, sTexture, NULL, &Rect ); 
    //SDL_FreeTexture(sTexture);
}








void sSDL::finishSDL(){
	std::cout << "Finish SDL..." << std::endl;
	if(initSDL){
		if( plotlegend){
			TTF_CloseFont(  font );
			TTF_CloseFont(  font_title );
		}
		SDL_DestroyTexture( texture);
		//SDL_FreeSurface( screen_surface);
		SDL_DestroyRenderer( renderer);
		SDL_DestroyWindow( window);
		if(!gpu) delete[]  cmap_rgba;
        
		if( plotlegend) TTF_Quit();
		SDL_Quit();
	}
}


bool sSDL::exitRequested() const{
	if(initSDL){
		SDL_PumpEvents();
		if (SDL_HasEvent(SDL_QUIT))
			return true;
		SDL_Event event;
		if (SDL_HasEvent(SDL_KEYDOWN)) {
			do SDL_PollEvent(&event); while(event.type != SDL_KEYDOWN);
			if (event.key.keysym.sym == SDLK_ESCAPE)
				return true;
		}
		return false;
	}
	return false;
}

void sSDL::checkEvent(double &elapsedtime, Timer &grosstime, double time){
	while (SDL_PollEvent(& e)) eventHandler(false, elapsedtime, grosstime, time);
}


void sSDL::save_image(std::string filename){
	std::cout << "Saving screenshot in file: " << filename << std::endl;
	SDL_Surface *sshot = SDL_CreateRGBSurface(0, screen_dimx, screen_dimy, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
	SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888, sshot->pixels, sshot->pitch);
	//SDL_SaveBMP(sshot, "screenshot.bmp"); //fromSDL2
	SDL_SavePNG(sshot, filename.c_str()); //from savepng.cpp
	SDL_FreeSurface(sshot);
	std::cout << "Saved screenshot..." << std::endl;
}


void sSDL::eventHandler(bool print, double &elapsedtime, Timer &grosstime, double time){
	switch( e.type) {
		case SDL_QUIT:
			if(print) printf( "Received SDL_QUIT - bye!\n");
			 cexit = true;
			return;

		case SDL_WINDOWEVENT:
			//printf( "SDL_WINDOWEVENT!!!\n");
			if(paused)
			if( e.window.event == SDL_WINDOWEVENT_RESIZED ||  e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED){
				if(print) printf( "SDL_WINDOWEVENT( SDL_WINDOWEVENT_RESIZED or SDL_WINDOWEVENT_SIZE_CHANGED detected), updating graphics in pause mode!!!\n");
				//updateScreen();//draw();
			}
			break;
		
		case SDL_MOUSEBUTTONUP:
			if(print) printf( "SDL_MOUSEBUTTONUP, button %d clicks %d\n",  e.button.button, (int) e.button.clicks);
			break;
		
		case SDL_MOUSEBUTTONDOWN:
			if(print) printf( "SDL_MOUSEBUTTONDOWN, button %d clicks %d\n",  e.button.button, (int) e.button.clicks);
			break;
		
		case SDL_KEYUP:
		case SDL_KEYDOWN:
			if( e.type == SDL_KEYUP){
				if(print) printf( "SDL_KEYUP: ");
			}
			else{
				if(print) printf( "SDL_KEYDOWN: ");
			}
		
			if(print) printf( "Keycode: %s (%d) Scancode: %s (%d)\n", 
				   SDL_GetKeyName( e.key.keysym.sym),  e.key.keysym.sym,
				   SDL_GetScancodeName( e.key.keysym.scancode),
				    e.key.keysym.scancode);
		
			if( e.key.keysym.sym == SDLK_ESCAPE) {
				if(print) printf( "You pressed ESC - bye!\n");
				 cexit = true;
				return;
			}
			if( e.key.keysym.sym == SDLK_q &&  e.type == SDL_KEYUP) {
				if(print) printf( "You pressed Q - bye!\n");
				 cexit = true;
				return;
			}
			if( e.key.keysym.sym == SDLK_s &&  e.type == SDL_KEYUP) {
				if(print) printf( "You pressed S - pause!\n");
				if(!paused){
					grosstime.stop();
					elapsedtime += grosstime.getElapsedTime();
				}
				std::string filename = "gross_" + ToString(nx) + "x" + ToString(ny) + "_" + ToString(time) + ".png";
				save_image(filename);
				if(!paused) grosstime.start();
				return;
			}
			if( e.key.keysym.sym == SDLK_p &&  e.type == SDL_KEYUP) {
				if(print) printf( "You pressed P - pause!\n");
				if(paused){
					paused = false; 
					grosstime.start();
				}
				else{
					paused = true;
					grosstime.stop();
					elapsedtime += grosstime.getElapsedTime();
				}
				return;
			}	
			if( e.key.keysym.sym == SDLK_UP &&  e.type == SDL_KEYUP) {
				if(print) printf( "You pressed UP key!\n");
				if( stepsUpScreen<10 &&  stepsUpScreen >= 2)  stepsUpScreen+=1;
				else  stepsUpScreen+=10;
				std::cout << " stepsUpScreen: " <<  stepsUpScreen << std::endl;
				return;
			}	
			if( e.key.keysym.sym == SDLK_DOWN &&  e.type == SDL_KEYUP) {
				if(print) printf( "You pressed DOWN key!\n");
				if( stepsUpScreen>12)  stepsUpScreen-=10;
				else if( stepsUpScreen<=12 &&  stepsUpScreen > 2)  stepsUpScreen-=1;
				
				std::cout << " stepsUpScreen: " <<  stepsUpScreen << std::endl;
				return;
			}	
			break;
		default:
			//printf( "SDL_Event of type: 0x%x received\n",  e.type);
			break;
	}
}





void sSDL::InitSDL(float factor_, size_t nx_, size_t ny_, float Lx_, float Ly_, bool gpu_){
using namespace std;
	std::cout << "Initializing SDL..." << std::endl;
	//SDL_Init(SDL_INIT_EVERYTHING);
	//SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
	if (SDL_Init(SDL_INIT_TIMER | SDL_INIT_VIDEO) < 0) {
		printf("Couldn't initialize SDL: %s", SDL_GetError());
		exit(1);
	}
	gpu = gpu_;
	nx = nx_;
	ny = ny_;
	Lx = Lx_;
	Ly = Ly_;
	plotlegend = true;
	factor = factor_;
	texXdim = nx * 2;
	texYdim = ny;
	posx = 0;
	posy = 0;
	bgcolor = (SDL_Color){245,255,250, 255};//{255,255,240, 255};

	bgcoloruint = ((uint)(bgcolor.a) << 24) | // convert colourmap to int
					((uint)(bgcolor.b) << 16) |
					((uint)(bgcolor.g) <<  8) |
					((uint)(bgcolor.r) <<  0);

	
	if( plotlegend){
		if(TTF_Init() < 0){
			printf("TTF_Init: %s\n", TTF_GetError());
			exit(1);;
		}
		 pts = 11;

		/* font_name = "arial.ttf";
		 font_name = "AppleGaramond-Bold.ttf";
		 font_name = "AppleGaramond.ttf";
		 font_name = "calibri.ttf";
		 font_name = "calibrib.ttf";*/
		 font_name = "arialbd.ttf";
		 font_title_name = "arialbd.ttf";
		
		 color = (SDL_Color){0, 0, 0, 255};


		 font = TTF_OpenFont( font_name.c_str(), ny* factor/24.);
		TTF_SetFontHinting(  font, TTF_STYLE_BOLD );
		string ss = "-2.2244e-10";
		 text_surface = My_TTF_Render( font,ss.c_str(), color);
		if(! text_surface){
			printf("TTF_RenderText_Solid: %s\n", TTF_GetError());
			exit(-5);
		}
		 text_legend_w =  text_surface->w;
		 text_legend_h =  text_surface->h;
		SDL_FreeSurface( text_surface);

		float step = 2.*Lx / ( pts-1);
		int leng = 0;
		for(int i = 0; i <  pts; i++){
			float val = -Lx + i * step;
			std::stringstream ss1;
			ss1.precision(3);
			//ss.setf(std::ios_base::scientific);
			ss1 << val;
			if(ss1.str().length() > leng){
				ss = ss1.str();
				leng = ss1.str().length();
			}
		}
		 text_surface = My_TTF_Render( font,ss.c_str(), color);
		if(! text_surface){
			printf("TTF_RenderText_Solid: %s\n", TTF_GetError());
			exit(-5);
		}
		 text_axis_w =  text_surface->w;
		 text_axis_h =  text_surface->h;
		SDL_FreeSurface( text_surface);


		
		 font_title = TTF_OpenFont( font_title_name.c_str(), ny* factor/18.);
		TTF_SetFontHinting(  font_title, TTF_STYLE_BOLD );
		ss = "Gross-Pitaevskii";
		 text_surface = My_TTF_Render( font,ss.c_str(), color);
		if(! text_surface){
			printf("TTF_RenderText_Solid: %s\n", TTF_GetError());
			exit(-5);
		}
		 text_title_h =  text_surface->h;
		SDL_FreeSurface( text_surface);



		barx = ( text_axis_h/4 +  text_axis_w +  text_axis_h) /  factor;
		bardim = nx/6.;
		posx = ( text_axis_w +  text_axis_h +  text_axis_h/4) / factor;
		posy = ( text_title_h*3  +  text_axis_h+ text_axis_h/4)/ factor;

		 texXdim = (posx + nx + barx + bardim + ( text_legend_w+ text_legend_w/10)/ factor ) * 2;


		 texXdim = (posx + nx + barx + bardim + ( text_legend_w+ text_legend_w/20)/ factor ) * 2;
		 texYdim = ny + posy + (2 *  text_axis_h +  text_axis_h/4)/ factor;
	}
	


	 screen_dimx =  factor *  texXdim;
	 screen_dimy =  factor *  texYdim;

	SDL_CreateWindowAndRenderer( screen_dimx,  screen_dimy, 0, & window, & renderer);
    SDL_SetWindowResizable(window, SDL_FALSE);

	//window = SDL_CreateWindow( "Server", screen_dimx,  screen_dimy, screen_dimx,  screen_dimy, SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE );
	//renderer = SDL_CreateRenderer( window, -1, SDL_RENDERER_ACCELERATED );
	//screen_surface = SDL_GetWindowSurface(  window );
	
	SDL_SetRenderDrawColor(  renderer, 255, 255, 255, 0xFF );
	texture = SDL_CreateTexture( renderer, SDL_PIXELFORMAT_BGR888, SDL_TEXTUREACCESS_STREAMING,  texXdim,  texYdim);


	std::string filename = "cmap.dat";
	std::ifstream filein(filename.c_str());
	if (!filein.is_open()){
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}
	filein >> ncol;
	 cmap_rgba = new uint[ncol]();


	float rcol,gcol,bcol;
	for (int i=0;i<ncol;i++){
		filein >> rcol >> gcol >> bcol;

		 cmap_rgba[i]=((int)(255.0f) << 24) | // convert colourmap to int
			((int)(bcol * 255.0f) << 16) |
			((int)(gcol * 255.0f) <<  8) |
			((int)(rcol * 255.0f) <<  0);
	}
	filein.close();    
 	initSDL = true;
}




void MakeTitle(SDL_Renderer* renderer, TTF_Font *font, SDL_Color textColor, const char *text, int y, int screen_dimx ) {
    SDL_Surface *text_surface = My_TTF_Render(font, text, textColor);
	if(!text_surface){
		printf("TTF_RenderText: %s\n", TTF_GetError());
		exit(-5);
	}
	SDL_Texture* sTexture = SDL_CreateTextureFromSurface( renderer, text_surface );
	

    SDL_Rect Rect;
	SDL_QueryTexture( sTexture, NULL, NULL, &Rect.w, &Rect.h );

	int posxx = 0;
	if( text_surface->w <  screen_dimx)
		posxx =  screen_dimx / 2 -  text_surface->w/2;
	Rect.x = posxx;
	Rect.y = y;
	SDL_RenderCopy( renderer, sTexture, NULL, &Rect ); 
    SDL_FreeSurface(text_surface);
}






void sSDL::updateScreen(float minvar0, float maxvar0, float minvar1, float maxvar1, float integral, float t, float t0){

	if( initSDL){
		char title[150];
	    if(gpu) snprintf( title, 150, "GPU Gross-Pitaevskii: abs(%.4f, %.4f) \
: arg(%.4f, %.4f) : int = %.4f : t = %.4f s : elapsedtime: %.2f s",\
 minvar0, maxvar0, minvar1, maxvar1, integral, t, t0);
	    else snprintf( title, 150, "CPU Gross-Pitaevskii: abs(%.4f, %.4f) \
: arg(%.4f, %.4f) : int = %.4f : t = %.4f s : elapsedtime: %.2f s",\
 minvar0, maxvar0, minvar1, maxvar1, integral, t, t0);

	    SDL_SetWindowTitle(  window, title );

		SDL_RenderClear( renderer);
		//plots
		SDL_RenderCopy( renderer,  texture, 0, 0); 

		if( plotlegend){
			//text
			Timer t1;
			if(getVerbosity() == DEBUG_VERBOSE) t1.start();
			int miny = 0;
			int maxy = ny *  factor;
			int posxx = ( posx + nx + barx+ bardim)* factor + text_axis_h/3;
			int posyy =  posy* factor -  text_axis_h*0.8;

			//text for left plot
			float step = (maxvar0-minvar0) / ( pts-1);
			float sdx = (maxy-miny)/( pts-1);
			for(int i = 0; i <  pts; i++){
				float val = maxvar0 - i * step;
				std::stringstream ss;
				ss.precision(2);
				ss.setf(std::ios_base::scientific);
				ss << val;

				//Render_Text( screen_surface,  font,  color, ss.str().c_str(), posxx, posyy + miny + i * (maxy-miny)/( pts-1));
                RendererText( renderer,  font,  color, ss.str().c_str(), posxx, posyy + miny + i * (maxy-miny)/( pts-1));
			}


			//text for right plot
			posxx +=  screen_dimx / 2;
			step = (maxvar1-minvar1) / ( pts-1);
			sdx = (maxy-miny)/( pts-1);
			for(int i = 0; i <  pts; i++){
				float val = maxvar1 - i * step;
				std::stringstream ss;
				//ss.precision(5);
				ss.precision(2);
				ss.setf(std::ios_base::scientific);
				ss << val;

				//Render_Text( screen_surface,  font,  color, ss.str().c_str(), posxx, posyy + miny + i * (maxy-miny)/( pts-1));
                RendererText( renderer,  font,  color, ss.str().c_str(), posxx, posyy + miny + i * (maxy-miny)/( pts-1));

			}

			//plot text in x axis
			posyy = (ny +  posy) *  factor + text_axis_h/4;
			posxx =  posx *  factor - text_axis_h/2;
			step = 2.*Lx / ( pts-1);
			sdx = (nx* factor)/( pts-1);
			for(int i = 0; i <  pts; i++){
				float val = -Lx + i * step;
				std::stringstream ss;
				ss.precision(3);
				//ss.setf(std::ios_base::scientific);
				ss << val;

				/*text_surface = My_TTF_Render( font,ss.str().c_str(), color);
				if(! text_surface){
					printf("TTF_RenderText_Solid: %s\n", TTF_GetError());
					exit(-5);
				}
				SDL_Rect dst = {posxx + i * sdx, posyy,  text_surface->w,  text_surface->h};
				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);
				dst.x +=  screen_dimx/2;
				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);

				dst.y =  posy* factor - text_legend_h;
				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);
				dst.x -=  screen_dimx/2;
				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);

				SDL_FreeSurface( text_surface);*/



                SDL_Surface *text_surface = My_TTF_Render(font, ss.str().c_str(), color);
	            if(!text_surface){
		            printf("TTF_RenderText: %s\n", TTF_GetError());
		            exit(-5);
	            }
	            SDL_Texture* sTexture = SDL_CreateTextureFromSurface( renderer, text_surface );
	            SDL_FreeSurface(text_surface);

                SDL_Rect dst;
	            SDL_QueryTexture( sTexture, NULL, NULL, &dst.w, &dst.h );

	            dst.x = posxx + i * sdx;
	            dst.y = posyy;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

                dst.x +=  screen_dimx/2;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

				dst.y =  posy* factor - text_legend_h;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

				dst.x -=  screen_dimx/2;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

			}
			//plot text in y axis
			posyy =  posy *  factor -  text_legend_h/2;
			posxx = (nx +  posx) *  factor + text_legend_h/4;

			step = 2.*Ly / ( pts-1);
			sdx = (ny* factor)/( pts-1);
			for(int i = 0; i <  pts; i++){
				float val = Ly - i * step;
				std::stringstream ss;
				ss.precision(3);
				//ss.setf(std::ios_base::scientific);
				ss << val;

				/*text_surface = My_TTF_Render( font,ss.str().c_str(), color);
				if(! text_surface){
					printf("TTF_RenderText_Solid: %s\n", TTF_GetError());
					exit(-5);
				}
				SDL_Rect dst = {posxx, posyy + i * sdx,  text_surface->w,  text_surface->h};

				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);
				dst.x +=  screen_dimx/2;
				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);

				dst.x =  posx* factor - text_surface->w -  text_legend_h/4;
				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);
				dst.x +=  screen_dimx/2;
				SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);

				SDL_FreeSurface( text_surface);*/



                SDL_Surface *text_surface = My_TTF_Render(font, ss.str().c_str(), color);
	            if(!text_surface){
		            printf("TTF_RenderText: %s\n", TTF_GetError());
		            exit(-5);
	            }
	            SDL_Texture* sTexture = SDL_CreateTextureFromSurface( renderer, text_surface );
	            SDL_FreeSurface(text_surface);

                SDL_Rect dst;
	            SDL_QueryTexture( sTexture, NULL, NULL, &dst.w, &dst.h );

	            dst.x = posxx;
	            dst.y = posyy + i * sdx;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

                dst.x +=  screen_dimx/2;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

				dst.x =  posx* factor - text_surface->w -  text_legend_h/4;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

				dst.x -=  screen_dimx/2;
	            SDL_RenderCopy( renderer, sTexture, NULL, &dst );

			}

			char title1[150];
			if(gpu) snprintf( title1, 150, "GPU Gross-Pitaevskii: %ux%u : t = %.4f s : elapsedtime: %.2f s", (uint)nx, (uint)ny, t, t0);
			else snprintf( title1, 150, "CPU Gross-Pitaevskii: %ux%u : t = %.4f s : elapsedtime: %.2f s", (uint)nx, (uint)ny, t, t0);

			/*text_surface = My_TTF_Render( font_title, title1,  color);
			if(! text_surface){
				printf("TTF_RenderText_Solid: %s\n", TTF_GetError());
				exit(-5);
			}
			posxx = 0;
			if( text_surface->w <  screen_dimx)
				posxx =  screen_dimx / 2 -  text_surface->w/2;

			SDL_Rect dst = {posxx,  text_title_h,  text_surface->w,  text_surface->h};
			SDL_BlitSurface( text_surface, NULL,  screen_surface, &dst);
			SDL_FreeSurface( text_surface);*/

            MakeTitle( renderer,  font_title,  color, title1, text_title_h, screen_dimx);


			if(getVerbosity() == DEBUG_VERBOSE){
				t1.stop();
				std::cout << "Time to render text:" << t1.getElapsedTimeInMilliSec() << " ms" << std::endl;
			}
		}
		//SDL_RenderClear( renderer);
		//SDL_RenderPresent( renderer);



	// Render the changes above
	SDL_RenderPresent( renderer);
	}
}
























