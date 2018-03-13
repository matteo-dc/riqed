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
    // number of lepton combos
    extern int combo_lep;
//// number of momenta
//extern int moms;
// number of equivalent momenta
extern int neqmoms;
// number of colours
extern int Nc;
// number of flavours
extern int Nf;
// string action
extern string action;
// number of betas
extern int nbeta;
// beta
extern vector<double> beta;
// beta label
extern vector<string> beta_label;
// number of theta
extern int ntheta;
// theta label
extern vector<string> theta_label;
    // kappa
    extern double kappa;
// number of sea masses
extern vector<int> nm_Sea;
// sea masses label
extern vector<vector<int>> SeaMasses_label;
    // sea masses label
    extern double mu_sea;
    // valence masses
    extern vector<double> mass_val;

        // effective masses
//        extern vvvd_t eff_mass;
        // deltam_cr
//        extern vvvd_t deltam_cr;

    // plaquette
    extern double plaquette;
    // coupling
//    extern double g2;
    // boosted coupling
//    extern double g2_tilde;

// inverse lattice spacing (Gev)
extern vector<double> ainv;
// Lambda QCD
extern double LambdaQCD;
// path to ensemble
extern string path_ensemble;

extern string path_to_ens;

// use Sigma1 parameter
extern int UseSigma1;
// use Effective Mass parameter
extern int UseEffMass;
    // N(p) vector
//    extern vector<int> Np;
    // path to mom list
    extern string mom_path;
//    // mom lists
//    extern vector<coords_t> mom_list;
//    extern vector<p_t> p, p_tilde;
//    extern vector<double> p2, p2_tilde;//, p2_tilde_eqmoms;
//    extern vector<double> p4, p4_tilde;
// string scheme
extern string scheme;
    // range for deltam_cr fit
    extern int delta_tmin;
    extern int delta_tmax;
// boundary conditions
//extern string BC_str;
extern string BC;
// minimum p2 value for continuum limit
extern double p2min;
// filtered yes/no
extern vector<bool> filt_moms;
// filter threshold
extern double thresh;
// compute MesLep
extern int compute_4f;
// out folder for quarks
extern string out_hadr;
// out folder for leptons
extern string out_lep;
// types of lepton propagators
extern int ntypes_lep;
// create only basic
extern int only_basic;

typedef enum {  
    LO = 0,
    EM = 1,
    P = 2
} ORDER;

typedef enum {
    QCD = 0,
    IN = 1,
    OUT = 2,
    M11 = 3,
    M22 = 4,
    M12 = 5,
    P11 = 6,
    P22 = 7
} MESLEP_TYPES;

extern ORDER ord;
extern MESLEP_TYPES meslep_t;
#endif
