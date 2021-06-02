#ifndef __CALCULATOR_H__
#define __CALCULATOR_H__

#include <iostream>


#include "complex.h"
#include "timer.h"

#include "RK1.h"
#include "RK2.h"
#include "RK3.h"
#include "init.h"
#include "draw.h"
#include "norm.h"

#include "constants.h"
#include "sdl.h"



template <class Real> 
class Calculator {
	private:
    bool gpu;
	Real Lx, Ly;
	size_t nx, ny;
	Real dx, dy,dt;
	Real dx2;
	Real dy2;
	Real oneoverdx;
	Real oneoverdy;
	Real oneoverdx2;
	Real oneoverdy2;


	Real t;
	Real gamma;
	Real omega;
	complex iomega;
	Real G;
	complex InvIminusGamma;
	complex scale;
	complex *a0_d, *a1_d, *a2_d;
	complex *a0, *a1, *a2;
	complex maxvar;
	complex minvar;
	Real integral;
	Real *d_sum;
	complex *d_max;
	complex *d_min;
	bool tempal;
	bool allocated;

	Timer grosstime;
	double elapsedtime;
	unsigned int iter;


	RK1<Real> mrk1;
	RK2<Real> mrk2;
	RK3<Real> mrk3;
	MapColor<Real> mapcolor;
	Normalize<Real> norm;


	complex *ptr;
	bool gptr;

	bool usetex;
	uint gpumode;




	void AllocateAndInit();
	Real xcoord(size_t i) const {	return (i - .5*nx) * dx;	}
	Real ycoord(size_t j) const {	return (j - .5*ny) * dy;	}
	void setScale() ;
	void updateTexture();

	void calc_cpu();
	void init_cpu();
	void norm_cpu();
	void init();


	void finishSDL();
	void step();


	bool randomInit;

	sSDL *sdl; 

	public:
	Calculator(Real Lx, Real Ly, Real dt, size_t nx, size_t ny, bool ingpu, uint gpumode_, bool randomInit_); // bool ingpu, true runs in GPU, false in CPU 
	void InitSDL(); 
	void InitSDL(Real factor);
	~Calculator();

	void setParam(Real gammai, Real omegai, Real Gi);
	void draw() ;
	void evolve(unsigned int numsteps, unsigned int stepsUpScreen_, bool savef) ;


	complex* GetPtr();

	Real getTime(){ return t;}


};

#endif
