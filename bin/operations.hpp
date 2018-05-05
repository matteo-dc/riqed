#ifndef OPER_HPP
#define OPER_HPP

#include <iostream>
#include "aliases.hpp"
#include "global.hpp"
#include <vector>
#include "read.hpp"
#include "read_input.hpp"

#ifndef EXTERN_OPER
 #define EXTERN_OPER extern
#endif

enum SCHEME_t{RI_MOM,SMOM,ERR};

EXTERN_OPER SCHEME_t sch;

struct oper_t
{
    vector<array<int,1>> linmoms;   // list of momenta used for Z, relative to glb list
    vector<array<int,3>> bilmoms;   // lists of momenta used for bilinears {k,i,j}
    vector<array<int,3>> meslepmoms; // list of momenta used for meslep
    
    void read_mom_list(const string &path);
    
    // volume
    double V;
    
    // size of nm and nr
    int _nm;
    int _nr;
    int _nmr;
    
    // size of linmoms and bilmoms
    int _linmoms;
    int _bilmoms;
    int _meslepmoms;
    
    // variables that characterize the data struct
    double _beta;
    string _beta_label;
    int _nm_Sea;
    string _SeaMasses_label;
    string _theta_label;
    double g2;
    double g2_tilde;
    
    // paths
    string ensemble_name;
    string path_to_ens;
    string path_to_beta;
    string path_to_moms;
    string path_print;
    
    // mom lists
    vector<coords_t> mom_list;
    vector<p_t> p, p_tilde;
    vector<double> p2, p2_tilde;//, p2_tilde_eqmoms;
    vector<double> p4, p4_tilde;
    vector<bool> filt_moms;

    // number of momenta
    int moms;
    
    // mPCAC
    void compute_mPCAC(const string &suffix);
    
    // deltamu and deltamcr
    vvd_t deltamu;
    vvd_t deltam_cr;
//    vvd_t read_deltam(const string path, const string name);
    void compute_deltam();
    void compute_deltam_from_prop();
    bool deltam_computed;
    
    // effective valence mass
    vvvd_t eff_mass;
    vvvd_t read_eff_mass(const string name);
    void compute_eff_mass();
    
    // effective valence mass (time dependent)
    vvvvd_t eff_mass_time;
    vvvvd_t read_eff_mass_time(const string name);
    
    // effective slope
    vd_t effective_slope(vd_t data, vd_t M, int TH);
    
    // effective sea mass
    vvvd_t eff_mass_sea;
    vvvd_t read_eff_mass_sea(const string name);
    void compute_eff_mass_sea();

    // compute the basic RC estimators
    void create_basic(const int b, const int th, const int msea);
    
    void set_ins();
    
    void set_moms();
    
    void set_ri_mom_moms();
    
    void set_smom_moms();
    
    void allocate();
    
    void resize_output(oper_t out);
    
    vector<string> setup_read_qprop(FILE* input[]);
    
    vector<string> setup_read_lprop(FILE* input_l[]);
    
    void build_prop(const vvvprop_t &prop, vvvprop_t &jprop);
    
    void ri_mom()
    {
        compute_sigmas();
        compute_bil();
        if(compute_4f) compute_meslep();
    }
    
    void smom()
    {
        ri_mom();
    }

    // create props
    void compute_prop();
    
    // definition of sigmas: trace of propagator
    vector<vvvvd_t> sigma;
    // compute sigmas
    void compute_sigmas();
    vvvd_t compute_sigma(vvvprop_t &jprop_inv, const int iproj, const int imom);
    
    // definition of projected bils
    vector<jproj_t> jG;
    // compute projected bils
    void compute_bil();

    // definition of projected meslep
    vector<jproj_meslep_t> jpr_meslep;
    // compute meslep
    void compute_meslep();
    
    // definition of jZq
    vector<jZq_t> jZq, jZq_EM;
    // compute Zq
    void compute_Zq();
    
    // definition of Z
    vector<jZbil_t> jZ, jZ_EM;
    // compute Zbils
    void compute_Zbil();
    
    // definition of Z (4fermions)
    vector<jZ4f_t> jZ_4f, jZ_4f_EM;
    // compute Z4f
    void compute_Z4f();
    
    // average r
    oper_t average_r(/*const bool recompute_Zbil=false*/) ;
    
    // chiral valence extrapolation
    oper_t chiral_extr();
    
    // O(g2a2) subtraction
    oper_t subtract();
    double subtraction(const int imom, const int ibil, const int LO_or_EM);
    double subtraction_q(const int imom, const int LO_or_EM);

    // evolution to 1/a scale
    oper_t evolve(const int b);
    
    // average of equivalent momenta
    oper_t average_equiv_moms();
    
    // plot Zq and Z
    void plot(const string suffix);
    
    // plot sigmas
    void plot_sigmas();    
};

// valarray of oper_t struct;
using voper_t=valarray<oper_t>;
using vvoper_t=valarray<voper_t>;
using vvvoper_t=valarray<vvoper_t>;

// chiral sea extrapolation
oper_t chiral_sea_extr(valarray<oper_t> in);

// theta average
oper_t theta_average(valarray<oper_t> in);

// a2p2 extrapolation
//void a2p2_extr();
voper_t a2p2_extr(voper_t in);

#endif

