#include <iostream>

#include <calculator.h>
#include <timer.h>

#include <omp.h>
#include "gnuplot.h"

template<class Real, bool gpu>
void Gross(){
  const int factor = 2;
	enum { nx = factor*128, ny = factor*128 };
	Real Lx = 10.0, Ly = 10.0;
	Real dx = 2.0*Lx /(Real)(nx - 1), dy = 2.0*Ly /(Real)(ny - 1);
	Real dt = 0.1 * dx * dy;
	std::cout << dt << std::endl;
	dt = 0.5 * ((dx*dx*dy*dy)/(dx*dx + dy*dy));
	std::cout << dt << std::endl;

	Real refactor = 4./Real(factor);

	Real sec = 1;
	Real totalsec = 50;
	int show = round(sec / dt);
	int totalit = round(totalsec / dt);
	std::cout << totalit << std::endl;
	std::cout << show << std::endl;

	Timer t0;
	t0.start();
	Calculator<Real> calc(Lx, Ly, dt, nx, ny, gpu, 0, true);
	calc.InitSDL(refactor);
	Real gamma = 0.1;
	Real omega = 0.85;
	Real G = 1000.0;
	calc.setParam(gamma, omega, G);
	//calc.draw();
	calc.evolve(totalit, show, true);
	t0.stop();
	std::cout << "Time: " << t0.getElapsedTime() << std::endl;

	//complex *a0 = calc.GetPtr();
	//tognuplot<Real>(a0, nx, ny, Lx, Ly, dx, dy, calc.getTime());
}

int main(){

	setVerbosity(DEBUG_VERBOSE);
	setVerbosity(VERBOSE);
	//setVerbosity(SILENT);

	
	const bool gpu = true;
	//GPU version
	Gross<float, gpu>();
	
	//omp_set_num_threads(4); 
	omp_set_num_threads(omp_get_max_threads());
	//CPU version
	Gross<float, false>();
	if(gpu) cudaDeviceReset();
}

