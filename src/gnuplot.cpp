

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>  
#include <cstring>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "complex.h"
#include "gnuplot.h"

using namespace std;  


template<class T>
inline std::string ToString(T number)
{
  std::stringstream ss;//create a stringstream
  ss << number;//add number to the stream
  return ss.str();//return a string with the contents of the stream
}


template<class Real>
void tognuplot(complex *a0, int Nx, int Ny, Real Lx, Real Ly, Real dx, Real dy, Real time){
	tognuplot(a0, Nx, Ny, Lx, Ly, dx, dy, time, 0);
	tognuplot(a0, Nx, Ny, Lx, Ly, dx, dy, time, 1);
}

template< class Real>
Real gettyp(complex a0, uint typ){
	Real val;
	switch(typ){
		case 0:
			val = a0.abs2();
		break;
		case 1:
			val = a0.arg();
		break;
		default:
			val = a0.abs2();
		break;
	}
	return val;
}

template<class Real>
void tognuplot(complex *a0, int Nx, int Ny, Real Lx, Real Ly, Real dx, Real dy, Real time, uint typ){

	string styp;
	switch(typ){
		case 0:
			styp = "abs2";
		break;
		case 1:
			styp = "arg";
		break;
		default:
			styp = "abs2";
		break;
	}



	string filenameini = "Res_" + ToString(Nx) + "x" + ToString(Ny) + "_" + ToString(time) + "_" + styp;
	string filename0 = filenameini + ".png";
	string filename = filenameini + ".dat";
	cout << "\nWriting File: " << filename << endl;
	ofstream fileout;
    fileout.open(filename.c_str(), ios::out);
    fileout.precision(14);
    fileout.setf(ios_base::scientific);
    if (!fileout.is_open()){
		cout << "Error opening file: " << filename << endl;
		exit(1);
    }
	Real min = 9999.;
	Real max = -9999;
    for(int j = 0; j < Ny; j++)
    for(int i = 0; i < Nx; i++){
        int id = i + j * Nx;
		Real val = gettyp<Real>(a0[id], typ);
		fileout << i * dx - Lx << "\t" << j * dx - Ly << "\t"  << val << endl;
        if ( fileout.fail() ) {
			cout << "Error saving file: " << filename << endl;
			exit(1);
        }
        if(i==Nx-1) {
			fileout << endl;
		    if ( fileout.fail() ) {
				cout << "Error saving file: " << filename << endl;
				exit(1);
		    }    
		}
		if( val > max ) max = val;
		if( val < min ) min = val;
    }
	fileout.close();


string gnup = "reset;\n\
text_title=\"t = " + ToString(time) + "\"\nfilein=\"";
gnup += ToString(filename) + "\"\n";
gnup += "fileout=\"" + ToString(filename0) + "\"\n\
set output\n\
sizein=1024\n\
a=" + ToString(Lx) +"\n\
b=" + ToString(Ly) +"\n\
d=(2*b)/(2*a)\n\
set view map\n\
unset surface\n\
set style data pm3d\n\
set style function pm3d\n\
set pm3d transparent\n\
set noclabel\n\
unset label\n\
set xrange [-a:a]\n\
set yrange [-b:b]\n\
#set xtics font \"Helvetica-Bold,12\"\n\
#set ytics font \"Helvetica-Bold,12\"\n\
#set cbrange [" + ToString(min) + ":" + ToString(max) + "]\n\
set cbrange [:]\n\
set format cb \"%1.2e\"\n\
set isosamples 3\n\
set size ratio d\n\
set xlabel \"x\" #font \"Helvetica-Bold,12\"\n\
set ylabel \"y\" #font \"Helvetica-Bold,12\"\n\
set palette gray\n\
set palette rgbformulae 30,-13,-23\n\
set size 0.75\n\
#set terminal postscript eps color enhanced\n\
#set terminal png transparent size 1024,1024\n\
set terminal png font arial 18 size sizein,sizein crop enhanced\n\
set out fileout\n\
set title text_title #font \"Helvetica-Bold,20\"\n\
splot filein using 1:2:3  notitle text_title with lines lc rgb \"black\"\n\
print filein";


    string filegnu = "gnu.script";
    ofstream fileout2(filegnu.c_str());
    //fileout.setf(ios::scientific);
    //fileout << setprecision(12);      
    fileout2 << gnup << endl;
    fileout2.close(); 

    string runp = "gnuplot " + filegnu;

    system(runp.c_str());

    runp = "\nconvert -trim " +  ToString(filename) + ".png" + " "  +  ToString(filenameini) + ".png";
    runp += "\nrm -f " + ToString(filename) + ".png";    
    //system(runp.c_str());
    
    runp = "\nrm -f " + ToString(filename);
    runp += "\nrm -f " + filegnu;
    system(runp.c_str());

    cout << "Image File: " << ToString(filenameini) << ".png" << endl;
}


template
void tognuplot<float>(complexs *a0, int Nx, int Ny, float Lx, float Ly, float dx, float dy, float time);
template
void tognuplot<double>(complexd *a0, int Nx, int Ny, double Lx, double Ly, double dx, double dy, double time);



