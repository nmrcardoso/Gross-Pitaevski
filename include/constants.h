#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__


#include <cuda.h>
#include <thrust/device_vector.h>
#include "cuda_common.h"
#include <cuda_runtime_api.h>
#include <sstream>
#include <string>
#include <cstring>


#include "complex.h"
#include "modes.h"


#define WARP_SIZE 32



Verbosity getVerbosity();

void setVerbosity(Verbosity verbosein);

template<class T>
inline std::string ToString(T number){
	std::stringstream ss;//create a stringstream
	ss << number;//add number to the stream
	return ss.str();//return a string with the contents of the stream
}


extern __constant__ bool UseTex;
extern __constant__ int nx;
extern __constant__ int ny;
extern __constant__ float omegas;
extern __constant__ float gammas;
extern __constant__ float gs;
extern __constant__ float dxs;
extern __constant__ float dxs2;
extern __constant__ float dys;
extern __constant__ float dys2;
extern __constant__ float oneoverdxs;
extern __constant__ double oneoverdxd;
extern __constant__ float oneoverdx2s;
extern __constant__ double oneoverdx2d;
extern __constant__ float oneoverdys;
extern __constant__ double oneoverdyd;
extern __constant__ float oneoverdy2s;
extern __constant__ double oneoverdy2d;
extern __constant__ float dts;
extern __constant__ float Lxs;
extern __constant__ float Lys;
extern __constant__ double omegad;
extern __constant__ double gd;
extern __constant__ double gammad;
extern __constant__ double dxd;
extern __constant__ double dxd2;
extern __constant__ double dyd;
extern __constant__ double dyd2;
extern __constant__ double dtd;
extern __constant__ double Lxd;
extern __constant__ double Lyd;
extern __constant__ complexs invIminusGammas;
extern __constant__ complexd invIminusGammad;
extern __constant__ complexs Iomegas;
extern __constant__ complexd Iomegad;




class tuneParam {
public:
	uint3 block;
	uint3 grid;
    unsigned int shared_bytes;
    std::string comment;
	double time;
	tuneParam(){};
	~tuneParam(){};
};



void copyToSymbol(bool usetex, int Nx, int Ny, float Omega, float Gamma, float G, \
	float dx, float dx2, float dy, float dy2, \
	float oneoverdx, float oneoverdx2, float oneoverdy, float oneoverdy2, float dt, complexs invIminusGamma, complexs iomega, float Lx, float Ly );

void copyToSymbol(bool usetex, int Nx, int Ny, double Omega, double Gamma, double G, \
	double dx, double dx2, double dy, double dy2, \
	double oneoverdx, double oneoverdx2, double oneoverdy, double oneoverdy2, double dt, complexd invIminusGamma, complexd iomega, double Lx, double Ly );



void Bind_UnBind_tex(complexs *a0, complexs *a1, complexs *a2, bool bind);
void Bind_UnBind_tex(complexd *a0, complexd *a1, complexd *a2, bool bind);


void Bind_UnBind_tex(complexs *a, int typ, bool bind);
void Bind_UnBind_tex(complexd *a, int typ, bool bind);





#endif

