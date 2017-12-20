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
    // lists of momenta used for bilinears {k,i,j}
    vector<array<int,3>> bilmoms;
    
    // compute the basic RC estimators
    void create_basic();
    
    void set_moms();
    
    void set_ri_mom_moms();
    
    void set_smom_moms();
    
    void resize_vectors();
    
    void ri_mom();

    void smom();

    // definition of jZq
    vector<jZ_t> jZq, jZq_em;
    vector<vvd_t> jZq_ave_r, jZq_em_ave_r;
    vector<vd_t> jZq_chir, jZq_em_chir;
    
    
    // create props and compute Zq
    void compute_prop();
    
    // definition of projected bils
    vector<jproj_t> jG_0, jG_em;
    vector<vvvd_t> jG_0_ave_r, jG_em_ave_r;
    vector<vvd_t> jG_0_chir, jG_em_chir;

    
    // compute projected bils
    void compute_bil();
    
    // definition of Z
    bool Zbil_computed{false};
    vector<jZbil_t> jZ, jZ_em;
    vector<vvd_t> jZ_chir, jZ_em_chir;
    
    // compute Zbils
    void compute_Zbil();
    
    // step string
    string step;
    
    // tuple
//    vector<Zq_tuple> Zq;
//    vector<G_tuple> G;
//    vector<Zbil_tuple> Zbil;
    
    // definition of the equivalent masses
    vector<double> m_eff_equivalent_Zq;
    vector<double> m_eff_equivalent;
    
    // average r
    oper_t average_r(/*const bool recompute_Zbil=false*/) ;
    
    // chiral extrapolation
    oper_t chiral_extr();
    
    // O(g2a2) subtraction
    oper_t subtract();

    
};

#endif

