#ifndef MESLEP_HPP
#define MESLEP_HPP

#ifndef EXTERN_MESLEP
 #define EXTERN_MESLEP extern
#endif

namespace meslep
{    
//    const int nmeslep = 10;
//    int nmeslep;
    const int nOp = 5;
    const int nGamma = 11; // number of independent combinations of gamma matrices
    
    const vector<size_t>         iG            = { 1, 2, 3, 4,   0,  10,11,12,13,14,15};
    const vector<int>            g5L_sign      = {-1,-1,-1,-1,  +1,  +1,+1,+1,+1,+1,+1};
    
    const vector<vector<size_t>> iG_of_iop     = {{0,1,2,3},{0,1,2,3},{4},{4},{5,6,7,8,9,10}};
    
    const vector<int>            g5_sign       = { -1,  +1,  -1,  +1,  +1};
    
    const vector<int> proj_norm = {4,4,1,1,24};
    const vector<int> op_norm = {1,1,1,1,2};
        
    enum kind{
        LO,     //  operator in pure QCD
        IN,     //  photon exchange between quark qIN  and charged lepton
        OUT,    //  photon exchange between quark qOUT and charged lepton
        M11,    //  em correction to the quark qIN  propagator
        M22,    //  em correction to the quark qOUT propagator
        M12,    //  photon exchanged between the two quarks
        P11,    //  pseudoscalar insertion on the quark qIN  propagator
        P22,    //  pseudoscalar insertion on the quark qOUT propagator
        S11,    //  scalar insertion on the quark qIN  propagator
        S22,    //  scalar insertion on the quark qOUT propagator
        QED=3
    };
}

namespace pr_meslep
{
    void set_ins();
    
    enum ins{LO,QED};
    
    EXTERN_MESLEP vector<ins> ins_list;
    EXTERN_MESLEP int nins;
}

namespace jmeslep
{
    void set_ins();

    enum ins{LO,IN,OUT,M11,M22,M12,P11,P22,S11,S22};
    
    EXTERN_MESLEP vector<ins> ins_list;
    EXTERN_MESLEP int nins;
    
    EXTERN_MESLEP int nLOampQED;

}

void build_meslep(const vvvprop_t &S1,const vvvprop_t &S2, const vvprop_t &L, valarray<jmeslep_t> &jmeslep);

jvproj_meslep_t compute_pr_meslep(vvvprop_t &jprop1_inv, valarray<jmeslep_t> &jmeslep, vvvprop_t  &jprop2_inv, vvd_t deltam_cr, vvd_t deltamu, const double q1, const double q2, const double ql);

#undef EXTERN_MESLEP

#endif