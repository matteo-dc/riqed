#ifndef OPER_HPP
#define OPER_HPP

#include <iostream>
#include "aliases.hpp"
#include "global.hpp"
#include <vector>

#ifndef EXTERN_OPER
 #define EXTERN_OPER extern
#endif

enum SCHEME_t{RI_MOM,SMOM,ERR};

EXTERN_OPER SCHEME_t sch;

struct oper_t
{
    vector<array<int,1>> linmoms;   // list of momenta used for Z, relative to glb list
    vector<array<int,3>> bilmoms;   // lists of momenta used for bilinears {k,i,j}
    
    // compute the basic RC estimators
    void create_basic();
    
    void set_moms();
    
    void set_ri_mom_moms();
    
    void set_smom_moms();
    
    void allocate();
    
    void resize_vectors(oper_t out);
    
    void ri_mom();

    void smom();

    // size of nm and nr
    int _nm;
    int _nr;
    int _nmr;
    
    // definition of jZq
    vector<jZ_t> jZq, jZq_em;
    
//    //!!TO BE CANCELLED
//    vector<vvd_t> jZq_ave_r, jZq_em_ave_r; //
//    vector<vd_t> jZq_chir, jZq_em_chir; //
//    vector<vd_t> jZq_sub, jZq_em_sub; //
//    vector<vd_t> jZq_evo, jZq_em_evo;     //
    
    // create props and compute Zq
    void compute_prop();
    
    // definition of projected bils
    vector<jproj_t> jG_0, jG_em;
    
//    //!!TO BE CANCELLED
//    vector<vvvd_t> jG_0_ave_r, jG_em_ave_r;
//    vector<vvd_t> jG_0_chir, jG_em_chir;
//    vector<vvd_t> jG_0_sub, jG_em_sub;
//    vector<vvd_t> jG_0_eqmoms, jG_em_eqmoms;

    
    // compute projected bils
    void compute_bil();
    
    // definition of Z
    bool Zbil_computed{false};
    vector<jZbil_t> jZ, jZ_em;
    
//    //!!TO BE CANCELLED
//    vector<vvd_t> jZ_chir, jZ_em_chir;
//    vector<vvd_t> jZ_sub, jZ_em_sub;
//    vector<vvd_t> jZ_evo, jZ_em_evo;
    
    // compute Zbils
    void compute_Zbil();
    
    // step string
    string step;
    
//    // definition of the equivalent masses
//    vvd_t m_eff_equivalent_Zq;
//    vvd_t m_eff_equivalent;
    
    // average r
    oper_t average_r(/*const bool recompute_Zbil=false*/) ;
    
    // chiral extrapolation
    oper_t chiral_extr();
    
    // O(g2a2) subtraction
    oper_t subtract();
    
    // evolution to 1/a scale
    oper_t evolve();
    
    // average of equivalent momenta
    oper_t average_equiv_moms();

    
};

void continuum_limit(oper_t out, const int LO_or_EM);

#endif

