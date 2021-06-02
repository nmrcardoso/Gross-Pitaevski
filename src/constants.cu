

#include "constants.h"


static Verbosity verbose = SILENT;

Verbosity getVerbosity(){
  return verbose;
}

void setVerbosity(Verbosity verbosein){
  verbose = verbosein;
}



__constant__ bool UseTex;
__constant__ int nx;
__constant__ int ny;
__constant__ float omegas;
__constant__ float gammas;
__constant__ float gs;
__constant__ float dxs;
__constant__ float dxs2;
__constant__ float dys;
__constant__ float dys2;
__constant__ double dxd;
__constant__ double dxd2;
__constant__ double dyd;
__constant__ double dyd2;



__constant__ float oneoverdxs;
__constant__ double oneoverdxd;
__constant__ float oneoverdx2s;
__constant__ double oneoverdx2d;
__constant__ float oneoverdys;
__constant__ double oneoverdyd;
__constant__ float oneoverdy2s;
__constant__ double oneoverdy2d;


__constant__ float dts;
__constant__ float Lxs;
__constant__ float Lys;
__constant__ double omegad;
__constant__ double gd;
__constant__ double gammad;
__constant__ double dtd;
__constant__ double Lxd;
__constant__ double Lyd;
__constant__ complexs invIminusGammas;
__constant__ complexd invIminusGammad;
__constant__ complexs Iomegas;
__constant__ complexd Iomegad;


texture<float2, 1, cudaReadModeElementType> texsp0;
texture<int4, 1, cudaReadModeElementType> texdp0;
texture<float2, 1, cudaReadModeElementType> texsp1;
texture<int4, 1, cudaReadModeElementType> texdp1;
texture<float2, 1, cudaReadModeElementType> texsp2;
texture<int4, 1, cudaReadModeElementType> texdp2;




void copyToSymbol(bool usetex, int Nx, int Ny, float Omega, float Gamma, float G, \
	float dx, float dx2, float dy, float dy2, \
	float oneoverdx, float oneoverdx2, float oneoverdy, float oneoverdy2, float dt, complexs invIminusGamma, complexs iomega, float Lx, float Ly ){
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( UseTex, &usetex, sizeof(bool) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( nx, &Nx, sizeof(int) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( ny, &Ny, sizeof(int) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( omegas, &Omega, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( gs, &G, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( gammas, &Gamma, sizeof(float) ));

    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dxs, &dx, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dxs2, &dx2, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dys, &dy, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dys2, &dy2, sizeof(float) ));

    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdxs, &oneoverdx, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdx2s, &oneoverdx2, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdys, &oneoverdy, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdy2s, &oneoverdy2, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dts, &dt, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( invIminusGammas, &invIminusGamma, sizeof(complexs) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( Iomegas, &iomega, sizeof(complexs) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( Lxs, &Lx, sizeof(float) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( Lys, &Ly, sizeof(float) ));

}
void copyToSymbol(bool usetex, int Nx, int Ny, double Omega, double Gamma, double G, \
	double dx, double dx2, double dy, double dy2, \
	double oneoverdx, double oneoverdx2, double oneoverdy, double oneoverdy2, double dt, complexd invIminusGamma, complexd iomega, double Lx, double Ly ){
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( UseTex, &usetex, sizeof(bool) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( nx, &Nx, sizeof(int) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( ny, &Ny, sizeof(int) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( omegad, &Omega, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( gd, &G, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( gammad, &Gamma, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dxd, &dx, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dxd2, &dx2, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dyd, &dy, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dyd2, &dy2, sizeof(double) ));

    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdxd, &oneoverdx, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdx2d, &oneoverdx2, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdyd, &oneoverdy, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( oneoverdy2d, &oneoverdy2, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( dtd, &dt, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( invIminusGammad, &invIminusGamma, sizeof(complexd) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( Iomegad, &iomega, sizeof(complexd) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( Lxd, &Lx, sizeof(double) ));
    CUDA_SAFE_CALL( cudaMemcpyToSymbol( Lyd, &Ly, sizeof(double) ));
}



void Bind_UnBind_tex(complexs *a0, complexs *a1, complexs *a2, bool bind){
  if(bind){
    CUDA_SAFE_CALL( cudaBindTexture(0, texsp0, a0));
    CUDA_SAFE_CALL( cudaBindTexture(0, texsp1, a1));
    CUDA_SAFE_CALL( cudaBindTexture(0, texsp2, a2));
    printf("Bind Texture for single precision\n");
  }
  else{
    CUDA_SAFE_CALL( cudaUnbindTexture(texsp0));
    CUDA_SAFE_CALL( cudaUnbindTexture(texsp1));
    CUDA_SAFE_CALL( cudaUnbindTexture(texsp2));
    printf("UnBind Texture for single precision\n");
  }
}
void Bind_UnBind_tex(complexd *a0, complexd *a1, complexd *a2, bool bind){
  if(bind){
    CUDA_SAFE_CALL( cudaBindTexture(0, texdp0, a0));
    CUDA_SAFE_CALL( cudaBindTexture(0, texdp1, a1));
    CUDA_SAFE_CALL( cudaBindTexture(0, texdp2, a2));
    printf("Bind Texture for double precision\n");
  }
  else{
    CUDA_SAFE_CALL( cudaUnbindTexture(texdp0));
    CUDA_SAFE_CALL( cudaUnbindTexture(texdp1));
    CUDA_SAFE_CALL( cudaUnbindTexture(texdp2));
    printf("UnBind Texture for double precision\n");
  }
}


void Bind_UnBind_tex(complexs *a, int typ, bool bind){
  if(bind){
	switch(typ){
		case 0:
			CUDA_SAFE_CALL( cudaBindTexture(0, texsp0, a));
		break;
		case 1:
			CUDA_SAFE_CALL( cudaBindTexture(0, texsp1, a));
		break;
		case 2:
			CUDA_SAFE_CALL( cudaBindTexture(0, texsp2, a));
		break;
	}
    //printf("Bind Texture for single precision\n");
  }
  else{
	switch(typ){
		case 0:
    		CUDA_SAFE_CALL( cudaUnbindTexture(texsp0));
		break;
		case 1:
    		CUDA_SAFE_CALL( cudaUnbindTexture(texsp1));
		break;
		case 2:
    		CUDA_SAFE_CALL( cudaUnbindTexture(texsp2));
		break;
	}
    //printf("UnBind Texture for single precision\n");
  }
}
void Bind_UnBind_tex(complexd *a, int typ, bool bind){
  if(bind){
	switch(typ){
		case 0:
    		CUDA_SAFE_CALL( cudaBindTexture(0, texdp0, a));
		break;
		case 1:
    		CUDA_SAFE_CALL( cudaBindTexture(0, texdp1, a));
		break;
		case 2:
    		CUDA_SAFE_CALL( cudaBindTexture(0, texdp2, a));
		break;
	}
    //printf("Bind Texture for double precision\n");
  }
  else{
	switch(typ){
		case 0:
    		CUDA_SAFE_CALL( cudaUnbindTexture(texdp0));
		break;
		case 1:
    		CUDA_SAFE_CALL( cudaUnbindTexture(texdp1));
		break;
		case 2:
    		CUDA_SAFE_CALL( cudaUnbindTexture(texdp2));
		break;
	}
    //printf("UnBind Texture for double precision\n");
  }
}





























