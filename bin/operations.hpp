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
    string ensamble_name;
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
    
    // deltam_cr
    vvvd_t deltam_cr;
    vvvd_t read_deltam_cr(const string name);
    void compute_deltam_cr();
    
    // effective valence mass
    vvvd_t eff_mass;
    vvvd_t read_eff_mass(const string name);
    void compute_eff_mass();
    
    // effective sea mass
    vvvd_t eff_mass_sea;
    vvvd_t read_eff_mass_sea(const string name);
    void compute_eff_mass_sea();


    // compute the basic RC estimators
    void create_basic(const int b, const int th, const int msea);
    
    void set_moms();
    
    void set_ri_mom_moms();
    
    void set_smom_moms();
    
    void allocate();
    
    void resize_output(oper_t out);
    
    vector<string> setup_read_qprop(FILE* input[]);
    
    vector<string> setup_read_lprop(FILE* input_l[]);
    
    void ri_mom();

    void smom();

    
    // definition of jZq
    vector<jZ_t> jZq, jZq_em;
    
    // create props
    void compute_prop();
    
    // compute Zq
    vvvd_t compute_jZq(vvvprop_t &jS_inv,const int imom);
    
    // definition of projected bils
    vector<jproj_t> jG_0, jG_em;
    
    // definition of projected charged bils (for 4fermions)
    vector<jproj_t> jG_0_4f, jG_em_4f;
    
    // definition of projected meslep
    vector<jproj_meslep_t> jpr_meslep_0, jpr_meslep_em;
    
    // compute projected bils
    void compute_bil();
    
    // compute meslep
    void compute_meslep();
    
    // definition of Z
    bool Zbil_computed{false};
    vector<jZbil_t> jZ, jZ_em;
    
    // definition of Z (4fermions)
    vector<jZ4f_t> jZ_4f, jZ_em_4f;
    
    // compute Zbils
    void compute_Zbil();
    
    // compute Z4f
    void compute_Z4f();
    
    
//  // step string
//    string step;
    
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

