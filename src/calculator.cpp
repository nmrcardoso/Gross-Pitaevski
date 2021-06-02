
#include <iostream>

#include <fstream>

#include <string>
#include <stdexcept>
#include <cstdio>


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include <typeinfo>


#include "complex.h"
#include "calculator.h"



#include "structs.h"
#include "savepng.h"
#include "device.h"


















template <class Real> 
void Calculator<Real>::AllocateAndInit(){

	if(gpu){
		size_t size = nx*ny * sizeof(complex);
		CUDA_SAFE_CALL(cudaMalloc((void**)&a0_d, size));
		CUDA_SAFE_CALL(cudaMalloc((void**)&a1_d, size));
		CUDA_SAFE_CALL(cudaMalloc((void**)&a2_d, size));
		CUDA_SAFE_CALL(cudaMemset(a0_d, 0, size));
		CUDA_SAFE_CALL(cudaMemset(a1_d, 0, size));
		CUDA_SAFE_CALL(cudaMemset(a2_d, 0, size));  	
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_sum, sizeof(Real)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_max, sizeof(complex)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_min, sizeof(complex)));
		if(usetex) Bind_UnBind_tex(a0_d, a1_d, a2_d, true);
	}
	else{
		a0 = new complex[nx*ny]();
		a1 = new complex[nx*ny]();
		a2 = new complex[nx*ny]();
	}

    allocated = true;

    InvIminusGamma = complex::one()/(complex::I()-gamma);
    //InvIminusGamma.print();
    iomega = complex::I() * omega;

	if(gpu){
		mrk1.setup(a0_d, a1_d, nx, ny);
		mrk2.setup(a0_d, a1_d, a2_d, nx, ny);
		mrk3.setup(a0_d, a1_d, a2_d, d_sum, nx, ny);
		norm.setup(a0_d, d_sum, d_min, d_max, nx, ny, dx, dy);
	}
	init();
}

template <class Real> 
Calculator<Real>::Calculator(Real Lx, Real Ly, Real dt, size_t nx, size_t ny, bool ingpu, uint gpumode_, bool randomInit_) : Lx(Lx), Ly(Ly), dt(dt),nx(nx), ny(ny), gpu(ingpu), gpumode(gpumode_), randomInit(randomInit_), dx(2.0*Lx/(nx-1)), dy(2.0*Ly/(ny-1)), t(0) {

	if(gpu){
		std::cout << "Running in GPU..." << std::endl;
		if(getVerbosity() >= VERBOSE) printGPUInformation();
		if( gpumode > 2 ) gpumode = 0;
		usetex = false;
		if(gpumode == 1 || gpumode == 2) usetex = true;
	}
	else std::cout << "Running in CPU..." << std::endl;
	gptr = false;
	sdl = new sSDL;
	sdl->cexit = false;
	sdl->initSDL = false;
	integral = 0.0;
	tempal = true;
	allocated = false;
	gamma = 0.1;
	omega = 0.85;
	G = 1000;
	dx2 = dx*dx;
	dy2 = dy*dy;

	oneoverdx = 1. / dx;
	oneoverdy = 1. / dy;
	oneoverdx2 = 1. / dx2;
	oneoverdy2 = 1. / dy2;

	sdl->paused = false;
	scale = 1;
	minvar = complex::zero();
	maxvar = complex::zero();
	grosstime.start();
	elapsedtime = 0.0;
	iter = 0;

	AllocateAndInit();	
}


template <class Real> 
Calculator<Real>::~Calculator() {	finishSDL();	}



template <class Real> 
complex* Calculator<Real>::GetPtr(){
	if(gpu){
		gptr = true;
		ptr = new complex[nx*ny];
		CUDA_SAFE_CALL(cudaMemcpy(ptr, a0_d, nx*ny * sizeof(complex),cudaMemcpyDeviceToHost));
		return ptr;
	}
	else return a0;
}


template <class Real> 
void Calculator<Real>::init(){
	if(gpu){
		copyToSymbol(false, nx, ny, omega, gamma, G, dx, dx2, dy, dy2, oneoverdx, oneoverdx2, oneoverdy, oneoverdy2, dt, InvIminusGamma, iomega, Lx, Ly);
		Init(a0_d, d_sum, nx, ny, randomInit);	
		norm.run(integral, minvar, maxvar, gpumode);
	}
	else	init_cpu();
}

template <class Real> 
void Calculator<Real>::setParam(Real gammai, Real omegai, Real Gi){
	gamma = gammai;
	omega = omegai;
	G = Gi;
    InvIminusGamma = complex::one()/(complex::I()-gamma);
    //InvIminusGamma.print();
    iomega = complex::I() * omega;

	init();
}




template <class Real> 
void Calculator<Real>::InitSDL(){
	InitSDL(1);
}

template <class Real> 
void Calculator<Real>::InitSDL(Real factor_){
using namespace std;

	sdl->InitSDL( (float)factor_, nx, ny, (float)Lx, (float)Ly, gpu);



	if(gpu){
		CUDA_SAFE_CALL(cudaMalloc((void **)& sdl->cmap_rgba_data, sizeof(uint)*sdl->ncol)); 
		CUDA_SAFE_CALL(cudaMemcpy(sdl->cmap_rgba_data, sdl->cmap_rgba, sizeof(uint)*sdl->ncol, cudaMemcpyHostToDevice));
		mapcolor.setup(sdl->cmap_rgba_data, sdl->ncol-1, nx, ny, sdl->posx, sdl->posy, sdl->barx, sdl->bardim, sdl->pts, sdl->plotlegend);
		delete[] sdl->cmap_rgba;
	}
}











template <class Real> 
void Calculator<Real>::step(){
	if(gpu){
		mrk1.run(gpumode);
		mrk2.run(gpumode);
		mrk3.run(gpumode);
		norm.run(integral, minvar, maxvar, gpumode);
	}
	else calc_cpu();
	t += dt;
}







template <class Real> 
void Calculator<Real>::updateTexture(){
	if(sdl->initSDL){
		void* raw_pixels;
		if(SDL_LockTexture(sdl->texture, 0, &raw_pixels, &sdl->pitch)){
			std::cout << "Failed to Lock Texture with \"SDL_LockTexture\"" << std::endl;
			finishSDL();
			std::cout << "Exiting.." << std::endl;
		}
		sdl->pixels = static_cast<unsigned int*>(raw_pixels);
		sdl->pitch1 = sdl->pitch/(sizeof(unsigned int));
		if( sdl->plotlegend && gpu == false) std::fill_n(sdl->pixels, sdl->pitch1*sdl->texYdim, sdl->bgcoloruint);
		if(gpu){
			if(tempal){
				std::fill_n(sdl->pixels, sdl->pitch1*sdl->texYdim, sdl->bgcoloruint);
				CUDA_SAFE_CALL(cudaMalloc((void**)&sdl->plot_rgba_data, sdl->pitch*sdl->texYdim));
				CUDA_SAFE_CALL(cudaMemcpy(sdl->plot_rgba_data, sdl->pixels, sdl->pitch*sdl->texYdim,cudaMemcpyHostToDevice));
				//CUDA_SAFE_CALL(cudaMemset(sdl->plot_rgba_data, 0, sdl->pitch*sdl->texYdim));
				tempal = false;
			}
			mapcolor.run(a0_d, sdl->plot_rgba_data, minvar, scale, sdl->pitch1);
			CUDA_SAFE_CALL(cudaMemcpy(sdl->pixels, sdl->plot_rgba_data, sdl->pitch*sdl->texYdim,cudaMemcpyDeviceToHost));
		}
		else {
			int i;
			#pragma omp parallel for private(i) collapse(2)
			for(int j = 0; j < ny; j++)
			for(i = 0; i < nx; i++){
				int i3d = sdl->posx + i + (sdl->posy + ny-1 -j) * sdl->pitch1;
				int id = i + j * nx;
				complex in = a0[id];
				Real frac = (in.abs2()-minvar.real()) * scale.real();
				int icol = (int)(frac * (Real)(sdl->ncol-1));
				if (icol < 0 || icol >= sdl->ncol) std::cout << i << "###" << j << "##" << frac << "##" << sdl->ncol << "###" << icol << std::endl;
				sdl->pixels[i3d] = sdl->cmap_rgba[icol];

				frac = (in.arg()-minvar.imag()) * scale.imag();
				icol = (int)(frac * (Real)(sdl->ncol-1));
				sdl->pixels[sdl->pitch1/2 + i3d] = sdl->cmap_rgba[icol];
			}

			if(sdl->plotlegend){
				#pragma omp parallel for private(i) collapse(2)
				for(int j = 0; j < ny; j++)
				for(i = 0; i < sdl->bardim; i++){
					int i2d = sdl->posx + nx + sdl->barx + i + (j + sdl->posy)*sdl->pitch1;
					Real frac = (Real)(ny-1 - j) / (Real)(ny-1);
					int icol = (int)(frac * (Real)(sdl->ncol-1));
					sdl->pixels[i2d] = sdl->cmap_rgba[icol]; 
					sdl->pixels[sdl->pitch1 / 2 + i2d] = sdl->cmap_rgba[icol]; 
				}




				int length = sdl->bardim/8;

				Real step = (Real)(ny-1)/(Real)(sdl->pts-1);
				
				#pragma omp parallel for private(i)
				for(i = 0; i < length; i++){
					for(int ii=0;ii<sdl->pts;ii++){
						int j = ii * step;
						int i2d = sdl->posx + nx + sdl->barx + i + (j + sdl->posy)*sdl->pitch1;
						sdl->pixels[i2d] = 0;
						sdl->pixels[sdl->pitch1 / 2 + i2d] = 0;
						sdl->pixels[i2d+sdl->bardim - length] = 0;
						sdl->pixels[sdl->pitch1 / 2 + i2d+sdl->bardim - length] = 0;
					}
				}


				for(i = 0; i < length; i++){
					for(uint ii=1;ii<sdl->pts-1;ii++){
						int j = ii * step;
						int i3d = sdl->posx + i + (sdl->posy + ny-1 -j) * sdl->pitch1;
						sdl->pixels[i3d] = 0;
						sdl->pixels[sdl->pitch1 / 2 + i3d] = 0;
						sdl->pixels[i3d + nx - length] = 0;
						sdl->pixels[sdl->pitch1 / 2 + nx - length + i3d] = 0;
					}	
				}

				step = (Real)(nx-1)/(Real)(sdl->pts-1);
				for(int j = 0; j < length; j++){
					for(uint ii=1;ii<sdl->pts-1;ii++){
						int i = ii * step;
						int i3d = sdl->posx + i + (sdl->posy + ny-1 -j) * sdl->pitch1;
						sdl->pixels[i3d] = 0;
						sdl->pixels[sdl->pitch1 / 2 + i3d] = 0;
						i3d = sdl->posx + i + (sdl->posy + j) * sdl->pitch1;
						sdl->pixels[i3d] = 0;
						sdl->pixels[sdl->pitch1 / 2 + i3d] = 0;
					}	
				}
			}
		}
		SDL_UnlockTexture(sdl->texture);
	}
}



template <class Real> 
void Calculator<Real>::setScale() {
	scale.real() = 1./(maxvar.real() - minvar.real());
	scale.imag() = 1./(maxvar.imag() - minvar.imag());
	if( isnan(scale.real()) || isinf(scale.real()) ) scale.real() = 0.0;
	if( isnan(scale.imag()) || isinf(scale.imag()) ) scale.imag() = 0.0;
}

template <class Real> 
void Calculator<Real>::finishSDL(){
	std::cout << "Free Resources..." << std::endl;
	if(allocated){
	    if(gpu){
			CUDA_SAFE_CALL(cudaFree(a0_d));
			CUDA_SAFE_CALL(cudaFree(a1_d));
			CUDA_SAFE_CALL(cudaFree(a2_d));
			CUDA_SAFE_CALL(cudaFree(d_sum));
			CUDA_SAFE_CALL(cudaFree(d_max));
			CUDA_SAFE_CALL(cudaFree(d_min));
			if(sdl->initSDL) CUDA_SAFE_CALL(cudaFree(sdl->cmap_rgba_data));
			if(gptr) delete[] ptr;
			if(usetex) Bind_UnBind_tex(a0, a1, a2, false);
        }
        else{
            delete[] a0;
            delete[] a1;
            delete[] a2;
        }
		if(!tempal && gpu && sdl->initSDL){
			CUDA_SAFE_CALL(cudaFree(sdl->plot_rgba_data));
			tempal = true;
		}
		allocated = false;
	}
	delete sdl;
}


template <class Real> 
void Calculator<Real>::draw() {	
	setScale();
	updateTexture();
	float t0 = (float)elapsedtime + grosstime.getElapsedTime();
	sdl->updateScreen((float)minvar.real(), (float)maxvar.real(), (float)minvar.imag(), (float)maxvar.imag(), (float)integral, (float)t, t0);
}

template <class Real> 
void Calculator<Real>::evolve(unsigned int numsteps, unsigned int stepsUpScreen_, bool savef) {
	grosstime.start();
	sdl->stepsUpScreen = stepsUpScreen_;
	for(unsigned int j = 1; j <= numsteps; ++j) {
		//std::cout << j << std::endl;
		step();
		iter++;
		if (j % sdl->stepsUpScreen == 1)
			draw();
		//if (sdl->exitRequested())
			//break;
		if(sdl->initSDL){ sdl->checkEvent(elapsedtime, grosstime, t);
			if(sdl->cexit) break;
			while(sdl->paused){	
				//sdl->checkEvent(elapsedtime, grosstime, t);
				SDL_WaitEvent(&sdl->e);
				sdl->eventHandler(false, elapsedtime, grosstime, t);
				if(sdl->cexit) break;
			}
			if(sdl->cexit) break;
		}
	}
	grosstime.stop();
	elapsedtime += grosstime.getElapsedTime();
	if(sdl->initSDL){
		draw();
		std::string filename = "gross_" + ToString(nx) + "x" + ToString(ny) + "_" + ToString(t) + ".png";
		if(savef)sdl->save_image(filename);

		if(0){
			sdl->paused = true;
			std::cout << "Finished evolve(), press p key to continue" << std::endl;
			while(sdl->paused){	
				SDL_WaitEvent(&sdl->e);
				sdl->eventHandler(false, elapsedtime, grosstime, t);
				if(sdl->cexit) break;
			}
		}
	}
	std::cout << "Elapsedtime: " << elapsedtime << std::endl;


}



template <class Real> 
void Calculator<Real>::init_cpu(){
	Real tmp = 0.0;
	int i;
	#pragma omp parallel for private(i) reduction(+:tmp)
	for(int j = 1; j < ny-1; j++)
	for(i = 1; i < nx-1; i++){
	    complex value = complex::one();
		if(randomInit) value = complex::make_complex(rand()/(Real)RAND_MAX, rand()/(Real)RAND_MAX);
		a0[i + j * nx] = value;
		tmp += value.abs2();
	}
	integral = tmp;
	norm_cpu();
}


template <class Real> 
Real max(Real a, Real b){
	if( a < b) return b;
	else return a;
}
template <class Real> 
Real min(Real a, Real b){
	if( a > b) return b;
	else return a;
}

template <class Real> 
void Calculator<Real>::norm_cpu(){
	integral = 1.0 / sqrt(integral * dx * dy); 
	//std::cout << "integral0=" << integral << std::endl;
	Real tmp = 0.0;
	Real max1 = 0.;
	Real max2 = 0.;
	Real min1 = 0.;
	Real min2 = 0.;
	int i;
	#pragma omp parallel for private(i) reduction(+:tmp) reduction(min:min1,min2) reduction(max:max1,max2)
	for(int j = 1; j < ny-1; j++)
	for(i = 1; i < nx-1; i++){
		int id = i + j * nx;        
		complex value = a0[id];
		value *= integral;
		a0[id] = value;
		tmp += value.abs2();
	  	//complex vals = complex::make_complex(value.abs2(), value.arg());
		max1 = max(max1, value.abs2());
		min1 = min(min1, value.abs2());
		max2 = max(max2, value.arg());
		min2 = min(min2, value.arg());

	}
	minvar = complex::make_complex(min1, min2);
	maxvar = complex::make_complex(max1, max2);
	integral = 1.0 / sqrt(tmp * dx * dy); 
	//std::cout << "integral1=" << integral << std::endl;
}






template <class Real> 
void Calculator<Real>::calc_cpu(){
	int i;
	#pragma omp parallel for private(i) collapse(2)
	for(int j = 1; j < ny-1; j++){
	for(i = 1; i < nx-1; i++){
		int id = i + j * nx;
		complex firstdx = a0[id + 1];
		complex res = a0[id - 1];
		complex actual = a0[id];
		complex seconddxy = (firstdx - actual * 2.0 + res) * 0.5 * oneoverdx2;
		firstdx = (firstdx - res) * 0.5 * oneoverdx;

		complex firstdy = a0[id + nx];
		res = a0[id - nx];
		seconddxy += (firstdy - actual * 2.0 + res) * 0.5 * oneoverdy2;
		firstdy  = (firstdy - res) * 0.5 * oneoverdy;

		Real px = i * dx - Lx;
		Real py = j * dy - Ly;

		firstdx = firstdy * px  - firstdx * py;
		px = ( (px * px + py * py) * 0.5) + (actual.abs2() * G);
		res =  actual * px;

		res = res - seconddxy - iomega * firstdx;;

		res = res * InvIminusGamma;            
		a1[id] = actual + res * dt;
	}
	}
	#pragma omp parallel for private(i) collapse(2)
	for(int j = 1; j < ny-1; j++){
	for(i = 1; i < nx-1; i++){
		int id = i + j * nx;
		complex firstdx = a1[id + 1];
		complex res = a1[id - 1];
		complex actual = a1[id];
		complex seconddxy = (firstdx - actual * 2.0 + res) * 0.5 * oneoverdx2;
		firstdx = (firstdx - res) * 0.5 * oneoverdx;

		complex firstdy = a1[id + nx];
		res = a1[id - nx];
		seconddxy += (firstdy - actual * 2.0 + res) * 0.5 * oneoverdy2;
		firstdy  = (firstdy - res) * 0.5 * oneoverdy;

		Real px = i * dx - Lx;
		Real py = j * dy - Ly;

		firstdx = firstdy * px  - firstdx * py;
		px = ( (px * px + py * py) * 0.5) + (actual.abs2() * G);
		res =  actual * px;

		res = res - seconddxy - iomega * firstdx;
		seconddxy = a0[id];

		res = res * InvIminusGamma;            
		a2[id] = seconddxy * 0.75 + actual * 0.25 + res * (dt * 0.25);
	}
	}
	minvar = complex::zero();
	maxvar = complex::zero();
	Real tmp = 0.0;
	#pragma omp parallel for private(i) reduction(+:tmp)
	for(int j = 1; j < ny-1; j++){
	for(i = 1; i < nx-1; i++){
		int id = i + j * nx;   
		complex firstdx = a2[id + 1];
		complex res = a2[id - 1];
		complex actual = a2[id];
		complex seconddxy = (firstdx - actual * 2.0 + res) * 0.5 * oneoverdx2;
		firstdx = (firstdx - res) * 0.5 * oneoverdx;

		complex firstdy = a2[id + nx];
		res = a2[id - nx];
		seconddxy += (firstdy - actual * 2.0 + res) * 0.5 * oneoverdy2;
		firstdy  = (firstdy - res) * 0.5 * oneoverdy;

		Real px = i * dx - Lx;
		Real py = j * dy - Ly;

		firstdx = firstdy * px  - firstdx * py;
		px = ( (px * px + py * py) * 0.5) + (actual.abs2() * G);
		res =  actual * px;

		res = res - seconddxy - iomega * firstdx;

		seconddxy = a0[id];
		res = res * InvIminusGamma; 
		px = 2.0 / 3.0;
		py = 1.0/ 3.0;         
		actual = seconddxy * py + actual * px + res * (dt * px);
		a0[id] = actual;
		tmp += actual.abs2();
	}
	}
	integral = tmp;
	norm_cpu();
}



template class Calculator<float>;
template class Calculator<double>;













































