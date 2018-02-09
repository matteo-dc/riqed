#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <omp.h>

//prop_t GAMMA=make_gamma();

//calculate the vertex function in a given configuration for the given equal momenta
prop_t make_vertex(const prop_t &prop1, const prop_t &prop2, const int mu)
{
    return prop1*GAMMA[mu]*GAMMA[5]*prop2.adjoint()*GAMMA[5];
}

//// compute the LO vertex
//jvert_t build_LO_vert(vvvprop_t &S1,vvvprop_t &S2, jvert_t &jVert_0)
//{
//#pragma omp parallel for collapse (4)
//    for(int ijack=0;ijack<njacks;ijack++)
//        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
//            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
//                for(int igam=0;igam<16;igam++)
//                    jVert_0[ijack][mr_fw][mr_bw][igam] += make_vertex(S1[ijack][0][mr_fw], S2[ijack][0][mr_bw],igam);
//    
//    return jVert_0;
//}
//
//// compute the EM vertex (up to 1st order in QED)
//jvert_t build_EM_vert(vvvprop_t &S1,vvvprop_t &S2,vvprop_t &S1_em,vvprop_t &S2_em,jvert_t &jVert_em)
//{
//#pragma omp parallel for collapse (4)
//    for(int ijack=0;ijack<njacks;ijack++)
//        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
//            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
//                for(int igam=0;igam<16;igam++)
//                    jVert_em[ijack][mr_fw][mr_bw][igam] += make_vertex(S1[ijack][0][mr_fw],S2_em[ijack][mr_bw],igam) + make_vertex(S1_em[ijack][mr_fw],S2[ijack][0][mr_bw],igam) + make_vertex(S1[ijack][1][mr_fw],S1[ijack][1][mr_bw],igam) ;
//    
//    return jVert_em;
//}

// compute LO and EM vertices
void build_vert(const vvvprop_t &S1,const vvvprop_t &S2/*,const vvprop_t &S1_em,const vvprop_t &S2_em*/,valarray<jvert_t> &jVert_LO_EM_P)
{
#pragma omp parallel for collapse (4)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int igam=0;igam<16;igam++)
                {
                    int r_fw = mr_fw % nr;
                    int r_bw = mr_bw % nr;
                    
                    jVert_LO_EM_P[LO][ijack][mr_fw][mr_bw][igam] += make_vertex(S1[ijack][0][mr_fw], S2[ijack][0][mr_bw],igam);
                    
                    jVert_LO_EM_P[EM][ijack][mr_fw][mr_bw][igam] += make_vertex(S1[ijack][0][mr_fw],S2[ijack][2][mr_bw],igam) + make_vertex(S1[ijack][0][mr_fw],S2[ijack][3][mr_bw],igam) + make_vertex(S1[ijack][2][mr_fw],S2[ijack][0][mr_bw],igam) + make_vertex(S1[ijack][3][mr_fw],S2[ijack][0][mr_bw],igam) + make_vertex(S1[ijack][1][mr_fw],S1[ijack][1][mr_bw],igam) ;
                    
                    // P (fw)
                    if(r_fw==0) jVert_LO_EM_P[2][ijack][mr_fw][mr_bw][igam] += make_vertex(+1.0*S1[ijack][4][mr_fw],S2[ijack][0][mr_bw],igam);
                    if(r_fw==1) jVert_LO_EM_P[2][ijack][mr_fw][mr_bw][igam] += make_vertex(-1.0*S1[ijack][4][mr_fw],S2[ijack][0][mr_bw],igam);
                    // P (bw)
                    if(r_bw==0) jVert_LO_EM_P[3][ijack][mr_fw][mr_bw][igam] += make_vertex(S1[ijack][0][mr_fw],+1.0*S2[ijack][4][mr_bw],igam);
                    if(r_bw==1) jVert_LO_EM_P[3][ijack][mr_fw][mr_bw][igam] += make_vertex(S1[ijack][0][mr_fw],-1.0*S2[ijack][4][mr_bw],igam);
                }
    
    // return jVert_LO_and_EM;
}


// invert the propagator
jprop_t invert_jprop(const jprop_t &jprop)
{
    jprop_t jprop_inv(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
    
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
            jprop_inv[ijack][mr]=jprop[ijack][mr].inverse();
    
    return jprop_inv;
}