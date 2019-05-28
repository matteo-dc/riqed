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
    void compute_deltam();
    void compute_deltam_from_prop();
    bool deltam_computed;
    
    // effective valence mass
    vvvd_t eff_mass;
    vvvd_t eff_mass_corr;
    vvvd_t read_eff_mass(const string name);
    vvvd_t read_eff_mass_corr(const string name);
    void compute_eff_mass();
    void compute_eff_mass_correction();
    
    // effective valence mass (time dependent)
    vvvvd_t eff_mass_time;
    vvvvd_t eff_mass_corr_time;
    vvvvd_t read_eff_mass_time(const string name);
    vvvvd_t read_eff_mass_corr_time(const string name);
    
    // effective slope
    vd_t effective_slope(vd_t data, vd_t M, int TH);
    
    // effective sea mass
    vd_t eff_mass_sea;
    vd_t read_eff_mass_sea(const string name);
    void compute_eff_mass_sea();

    // compute the basic RC estimators
    void create_basic(const int b, const int th, const int msea);
    
    void set_ins();
    
    void set_moms();
    
    void set_ri_mom_moms();
    
    void set_smom_moms();
    
    void clear_all();
    
    // allocate vectors
    void allocate();
    // allocate valarrays
    void allocate_val();
    // check allocation
    void check_allocation();
    
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
    
    // print averaged quantities
    void print(const string suffix);
    
    // print extrapolated Z
    void printZ(const string suffix);
    
    // load averaged quantities and extrapolated Z
    void load(const string suffix);
    
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
    // definition of ZV/ZA and ZP/ZS
    vector<jZbil_t> jZVoverZA;
    vector<jZbil_t> jZPoverZS;
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
    
    oper_t subOa2(const int b);


    // evolution to 1/a scale
//    oper_t evolve(const int b);
    oper_t evolveToAinv(const double ainv);
    
    // evolution to 1/a scale of mixed eta
    oper_t evolve_mixed(double ainv);
    
    // average of equivalent momenta
    oper_t average_equiv_moms();
    
    // returns the range in which x is contained
    pair<double,double> get_a2p2tilde_range(const int size,const double a2p2_ref,const size_t n=5) const;
    pair<double,double> get_a2p2_range(const int size,const double a2p2_ref,const size_t n=5) const;
    
    // interpolate to p2ref
    oper_t interpolate_to_p2ref(int b);
    vector<jZq_t> interpolate_to_p2ref_Zq(const double a2p2_ref,const int LO_or_EM);
    vector<jZbil_t> interpolate_to_p2ref_Zbil(const double a2p2_ref,const int LO_or_EM);
    vector<jZ4f_t> interpolate_to_p2ref_Z4f(const double a2p2_ref,const int LO_or_EM);
    
    // interpolate to p2=1/a2
    oper_t interpolate_to_ainv(int b);
    
    // plot Zq and Z
    void plot(const string suffix);
    void plot(const string suffix, int b);
    // plot chiral extrapolations
    void plot_bil_chir_extr(int mom, int ins, int ibil, vd_t x, vd_t y, vd_t dy, vvd_t jpars);
    void plot_meslep_chir_extr(int mom, int ins, int iop1, int iop2, vd_t x, vd_t y, vd_t dy, vvd_t jpars);
    
    // plot sigmas
    void plot_sigmas();
    
    // a2p2->0 extrapolation
    oper_t a2p2_extr();
};

// valarray of oper_t struct;
using voper_t=valarray<oper_t>;
using vvoper_t=valarray<voper_t>;
using vvvoper_t=valarray<vvoper_t>;
using vvvvoper_t=valarray<vvvoper_t>;
using vvvvvoper_t=valarray<vvvvoper_t>;

// chiral sea extrapolation
oper_t chiral_sea_extr(valarray<oper_t> in);

// theta average
voper_t theta_average(vvoper_t in);
//oper_t theta_average(valarray<oper_t> in);

// difference between interacting and free theory
oper_t compute_eta(voper_t oper_for_eta);
oper_t compute_eta_uncorr(voper_t oper_for_eta1,voper_t oper_for_eta2);

// a2p2->0 extrapolation combined on betas
voper_t a2p2_extr_combined_on_betas(voper_t in);

// combined chiral sea extrapolation
voper_t combined_chiral_sea_extr(vvoper_t in);

// test tree level projection
void test_gamma();

#endif

