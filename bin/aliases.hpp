#ifndef ALIASES_HPP
#define ALIASES_HPP

#ifdef HAVE_CONFIG_H
#include <config.hpp>
#endif

#ifndef EIGEN_H
#include <Eigen/Dense>
#define EIGEN_H
#endif

#include <valarray>
#include <complex>
#include <vector>
#include <tuple>

using namespace std;
using namespace Eigen;

//coordinates in the lattice
using coords_t=array<int,4>;

//p components in the lattice
using p_t=array<double,4>;

//complex double
using dcompl=complex<double>;

//propagator (12X12)
using prop_t=Matrix<dcompl,12,12>;

//list of propagators
using vprop_t=valarray<prop_t>;
using vvprop_t=valarray< vprop_t >;
using vvvprop_t=valarray< vvprop_t >;
using vvvvprop_t = valarray< vvvprop_t >;

//list of gamma for a given momentum
using qline_t=valarray<prop_t>;
using vqline_t=valarray<qline_t>;
using vvqline_t=valarray<vqline_t>;
using vert_t = vvqline_t;

//list of jackknife propagators
using jprop_t=vvvprop_t;

//list of jackknife vertices
using jvert_t=valarray< vert_t > ;

//valarray of complex double
using vd_t=valarray<double>;

//valarray of valarray of complex double
using vvd_t=valarray< valarray<double> > ;

//valarray of valarray of valarray of complex double
using vvvd_t=valarray< valarray< valarray<double> > >;
using vvvvd_t=valarray<vvvd_t>;
using vvvvvd_t=valarray<vvvvd_t>;
using vvvvvvd_t=valarray<vvvvvd_t>;

//valarray of complex double
using vdcompl_t=valarray<dcompl>;
using vvdcompl_t=valarray< vdcompl_t >;
using vvvdcompl_t=valarray< vvdcompl_t >;
using vvvvdcompl_t=valarray< vvvdcompl_t >;

//valarray of Eigen Vectors
using vXd_t=valarray<VectorXd>;
using vvXd_t=valarray<vXd_t>;
using vvvXd_t=valarray<vvXd_t>;

//valarray of Eigen Matrix (5x5)
using O4f_t=Matrix<double,5,5>;

//useful notation
using jproj_t=vvvvvd_t;
using jmeslep_t=valarray<jvert_t>;
using jvproj_meslep_t = valarray<vvvvvd_t> ;
using jproj_meslep_t = valarray< vvvvvd_t >;
using jZq_t=vvd_t;
using jZbil_t=vvvvd_t;
using jZ4f_t =vvvvvd_t;

//tuple
using Zbil_tup=tuple<vvvvd_t,vvvvd_t>;
using Zq_tup=tuple<vvd_t,vvd_t>;
using sigma_tup=tuple<vvvvd_t,vvvvd_t>;
using Zmeslep_tup=tuple<vvvvvd_t,vvvvvd_t>;
using Z4f_tup=Zmeslep_tup;

#endif
