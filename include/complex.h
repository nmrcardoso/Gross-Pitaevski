#ifndef COMPLEX_H
#define COMPLEX_H


#include <string.h>
#include <iostream>
#include <iomanip>

#include <math.h>

#include <cuda_common.h>
#include <cuda_vector_types.h>




/**
  @brief Class declaration for complex numbers in single and double precision
*/
template <class Real> 
class _complex {

 public:
    /** 
      @brief complex number representation, float2/double2 for single and double precision respectively.
      val.x hold real part and val.y hold the imaginary part
    */
    typedef typename MakeVector<Real, 2>::type Real2;
    Real2 val;

  M_HOSTDEVICE _complex(){
#ifndef __CUDA_ARCH__
    val.x = (Real)0.0; val.y = (Real)0.0;
#endif
  }
  M_HOSTDEVICE _complex(const Real REF(a), const Real REF(b)){
    val.x = a; val.y = b;
  }


  M_HOSTDEVICE _complex<Real>& operator=(const Real REF(a)) {
    val.x = a; val.y = 0;
    return *this;
  };
  M_HOSTDEVICE _complex<Real>& operator=(const  _complex<Real> REF(a)) {
    val.x = a.val.x; val.y = a.val.y;
    return *this;
  };

  // assignment of a pair of Ts to complex
  M_HOSTDEVICE _complex<Real>& operator=(const Real ARRAYREF(a,2)) {
    val.x = a[0]; val.y = a[1];
    return *this;
  };

  M_HOSTDEVICE bool operator==(const _complex<Real> REF(a)) const {
    if(val.x == a.val.x && val.y == a.val.y) return true;
    return false;
  };
  M_HOSTDEVICE bool operator!=(const _complex<Real> REF(a)) const {
    if(val.x == a.val.x && val.y == a.val.y) return false;
    return true;
  };

  // return references to the T and imaginary components
  M_HOSTDEVICE Real& real() {return val.x;};
  M_HOSTDEVICE Real& imag() {return val.y;};


  M_HOST bool operator==( const _complex<Real> &A ) {
    _complex<Real> B = *this;
    return ! memcmp( &A, &B, sizeof(_complex<Real>) );
  }
  M_HOST bool operator!=( const _complex<Real> &A ) {
    _complex<Real> B = *this;
    return memcmp( &A, &B, sizeof(_complex<Real>) );
  }

  // add complex numbers
  M_HOSTDEVICE _complex<Real> operator+(const _complex<Real> REF(b)) const {
    return _complex(val.x+b.val.x, val.y+b.val.y);
  }


  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator+(const Real REF(b)) const {
    return _complex(val.x+b, val.y);
  }
 
  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator+=(const Real REF(b)) {
    val.x += b;
    return *this;
  } 
  // add complex numbers
  M_HOSTDEVICE _complex<Real> operator+=(const _complex<Real> REF(b)) {
    val.x += b.val.x;
    val.y += b.val.y;
    return *this;
  }


  // subtract complex numbers
  M_HOSTDEVICE _complex<Real> operator-(const _complex<Real> REF(b)) const {
    _complex<Real> result;
    result.val.x = val.x - b.val.x;
    result.val.y = val.y  - b.val.y;
    return result;
  }

  // negate a complex number
  M_HOSTDEVICE _complex<Real> operator-() const {
    _complex<Real> result;
    result.val.x = -val.x;
    result.val.y = -val.y;
    return result;
  }

  // subtract scalar from complex
  M_HOSTDEVICE _complex<Real> operator-(const Real REF(b)) const {
    return  _complex(val.x-b,val.y);
  }

  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator-=(const Real REF(b)) {
    val.x -= b;
    return *this;
  } 
  // add complex numbers
  M_HOSTDEVICE _complex<Real> operator-=(const _complex<Real> REF(b)) {
    val.x -= b.val.x;
    val.y -= b.val.y;
    return *this;
  }
  // multiply complex numbers
  M_HOSTDEVICE _complex<Real> operator*(const _complex<Real> REF(b)) const {
    return _complex(val.x * b.val.x - val.y * b.val.y, val.y * b.val.x + val.x * b.val.y);
  }
  // multiply complex numbers
  M_HOSTDEVICE _complex<Real> operator*=(const _complex<Real> REF(b)) {
    Real tmp = val.x * b.val.x - val.y * b.val.y;
    val.y = val.y * b.val.x + val.x * b.val.y;
    val.x = tmp;
    return *this;
  }

  // multiply complex with scalar
  M_HOSTDEVICE _complex<Real> operator*(const Real REF(b)) const {
    return _complex(val.x * b, val.y * b);
  }
 
  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator*=(const Real REF(b)) {
    val.x *= b;
    val.y *= b;
    return *this;
  } 

  // divide complex numbers
  M_HOSTDEVICE _complex<Real> operator/(const _complex<Real> REF(b)) const {
    Real tmp = (Real)1.0 / ( b.val.x * b.val.x + b.val.y * b.val.y );
    _complex<Real> result;
    result.val.x = (val.x * b.val.x + val.y * b.val.y ) * tmp;
    result.val.y = (val.y * b.val.x - val.x * b.val.y ) * tmp;
    return result;
  }


  // divide complex by scalar
  M_HOSTDEVICE _complex<Real> operator/(const Real REF(b)) const {
    return _complex(val.x /b, val.y/b);
  }

  M_HOSTDEVICE _complex<Real> operator/=(const Real REF(b)) {
    val.x /= b;
    val.y /= b;
    return *this;
  }






  M_HOSTDEVICE    volatile _complex<Real>& operator +=( volatile  _complex<Real> & a ) volatile{
    val.x += a.val.x;
    val.y += a.val.y;
    return *this;
  }






  // complex conjugate
  M_HOSTDEVICE _complex<Real> operator~() const {
    return _complex(val.x, -val.y);
  }
  // complex conjugate
  M_HOSTDEVICE _complex<Real> conj() const {
    return _complex(val.x, -val.y);
  }

  // complex modulus (complex absolute)
  M_HOSTDEVICE Real abs() const {
    return sqrt( val.x*val.x + val.y*val.y );
  }
  // squaval.x complex modulus
  M_HOSTDEVICE Real abs2() const {
    return  ( val.x*val.x + val.y*val.y );
  }

  // complex phase angle
  M_HOSTDEVICE Real phase() const {
    return atan2( val.y, val.x );
  }

  M_HOSTDEVICE Real angle() const {
    return atan2( val.y, val.x );
  }

  // arg
  M_HOSTDEVICE Real arg() const {
    return atan2( val.y, val.x);
  }
/*  M_HOSTDEVICE Real arg() const {
#ifndef __CUDA_ARCH__
    return atan2( val.y, val.x);;
#else
    Real r = sqrt( val.x*val.x + val.y*val.y );
    Real res = acos(val.x /r);
    if(val.y < 0)
      res = -res;
    return res;
#endif
  }*/
  
  
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complex(const Real REF(a), const Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const volatile Real REF(a), const volatile Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const Real REF(a), const volatile Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const volatile Real REF(a), const Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const Real REF(a), const Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }



  // return constant number one
  static M_HOSTDEVICE  _complex<Real> one() {
    return make_complex((Real)1.0, (Real)0.0);
  }
  // return constant number one
  static M_HOSTDEVICE  _complex<Real> unit() {
    return make_complex((Real)1.0, (Real)0.0);
  }

  // return constant number zero
  static M_HOSTDEVICE  _complex<Real> zero() {
    return make_complex((Real)0.0, (Real)0.0);
  }


  // return constant number I
  static M_HOSTDEVICE  _complex<Real> I() {
    return make_complex((Real)0.0, (Real)1.0);
  }

  // print matrix contents 
  M_HOSTDEVICE void print() {
    printf("%.10e + %.10ej\n", val.x, val.y);
  }

    
  friend M_HOST std::ostream& operator<<( std::ostream& out, _complex<Real> M ) {
    //cout << std::scientific;
    //out << std::setprecision(14);
    out << M.real() << '\t' << M.imag();
    return out;
  }
};


//
// Define the common complex types
//
#define	complex		_complex<Real>
typedef _complex< float> complexs;
typedef _complex< double> complexd;


// a possible alternative to a single complex constructor
static M_HOSTDEVICE complexs make_complexs(float a, float b){
  complexs res;
  res.real() = a;
  res.imag() = b;
  return res;
}

// a possible alternative to a single complex constructor
static M_HOSTDEVICE complexs make_complexs(float2 a){
  complexs res;
  res.real() = a.x;
  res.imag() = a.y;
  return res;
}


// a possible alternative to a double complex constructor
static M_HOSTDEVICE complexd make_complexd(double a, double b){
  complexd res;
  res.real() = a;
  res.imag() = b;
  return res;
}

static M_HOSTDEVICE complexd make_complexd(double2 a){
  complexd res;
  res.real() = a.x;
  res.imag() = a.y;
  return res;
}





#endif // #ifndef COMPLEX_H
