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
    
    void resize_output(oper_t out);
    
    void ri_mom();

    void smom();

    // size of nm and nr
    int _nm;
    int _nr;
    int _nmr;
    
    // size of linmoms and bilmoms
    int _linmoms;
    int _bilmoms;
    
    // definition of jZq
    vector<jZ_t> jZq, jZq_em;
    
    // create props and compute Zq
    void compute_prop();
    
    // definition of projected bils
    vector<jproj_t> jG_0, jG_em;
    
    // compute projected bils
    void compute_bil();
    
    // definition of Z
    bool Zbil_computed{false};
    vector<jZbil_t> jZ, jZ_em;
    
    // compute Zbils
    void compute_Zbil();
    
    // step string
    string step;
    
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
    
    // plot Zq and Z
    void plot(const string suffix);
    
};

void continuum_limit(oper_t out, const int LO_or_EM);

#endif

