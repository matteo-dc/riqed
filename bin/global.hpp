#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include "aliases.hpp"
#include <string>
#include <fstream>
#include <omp.h>

// number of configurations
extern int nconfs;
// number of jackknives
extern int njacks;
// size of the cluster
extern int clust_size;
// sizes
extern coords_t size;
// confs range
extern int conf_init, conf_step;
// number of valence masses
extern int nm;
// number of equivalent bilinear mass combinations
extern int neq;
// number of equivalent Zq mass combinations
extern int neq2;
// number of bilinears
extern int nbil;
// number of Wilson parameters
extern int nr;
// number of mr combinations
extern int nmr;
// number of types
extern int ntypes;
// number of hits
extern int nhits;
// number of combos
extern int combo;
// number of momenta
extern int moms;
// number of equivalent momenta
extern int neqmoms;
// number of colours
extern int Nc;
// number of flavours
extern int Nf;
// string action
extern string action;
// beta
extern double beta;
// kappa
extern double kappa;
// sea mass
extern double mu_sea;
// valence masses
extern vector<double> mass_val;
// effective masses
extern vvd_t eff_mass;
// deltam_cr
extern vvvd_t deltam_cr;
// plaquette
extern double plaquette;
// coupling
extern double g2;
// boosted coupling
extern double g2_tilde;
// inverse lattice spacing (Gev)
extern double ainv;
// Lambda QCD
extern double LambdaQCD;
// path to ensemble
extern string path_ensemble_str;
// use Sigma1 parameter
extern int UseSigma1;
// use Effective Mass parameter
extern int UseEffMass;
// N(p) vector
extern vector<int> Np;
// path to mom list
extern string mom_path;
// mom lists
extern vector<coords_t> mom_list;
extern vector<p_t> p, p_tilde;
extern vector<double> p2, p2_tilde;
extern vector<double> p4, p4_tilde;
// string scheme
extern string scheme;
// range for deltam_cr fit
extern int delta_tmin;
extern int delta_tmax;
// boundary conditions
extern string BC_str;
// minimum p2 value for continuum limit
extern double p2min;

typedef enum {  
    LO = 0,
    EM = 1
} ORDER;

extern ORDER ord;

#endif
