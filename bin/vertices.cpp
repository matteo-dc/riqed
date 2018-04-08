#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include "contractions.hpp"
#include <omp.h>

//calculate the vertex function in a given configuration for the given equal momenta
prop_t make_vertex(const prop_t &prop1, const prop_t &prop2, const int mu)
{
    return prop1*GAMMA[mu]*GAMMA[5]*prop2.adjoint()*GAMMA[5];
}

// compute LO and EM vertices
void build_vert(const vvvprop_t &S1,const vvvprop_t &S2,valarray<jvert_t> &jVert_LO_EM_P_S)
{
#pragma omp parallel for collapse (4)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int igam=0;igam<16;igam++)
                {
                    // LO
                    jVert_LO_EM_P_S[LO][ijack][mr_fw][mr_bw][igam] +=
                        make_vertex(S1[ijack][_LO][mr_fw], S2[ijack][_LO][mr_bw],igam);
                    
                    // EM: Self + Tadpole + Exchange
                    jVert_LO_EM_P_S[EM][ijack][mr_fw][mr_bw][igam] +=
                        make_vertex(S1[ijack][_LO][mr_fw],S2[ijack][_FF][mr_bw],igam) +
                        make_vertex(S1[ijack][_LO][mr_fw],S2[ijack][_T ][mr_bw],igam) +
                        make_vertex(S1[ijack][_FF][mr_fw],S2[ijack][_LO][mr_bw],igam) +
                        make_vertex(S1[ijack][_T ][mr_fw],S2[ijack][_LO][mr_bw],igam) +
                        make_vertex(S1[ijack][_F ][mr_fw],S1[ijack][_F ][mr_bw],igam) ;
                    
                    // Pfw
                    jVert_LO_EM_P_S[Pfw][ijack][mr_fw][mr_bw][igam] +=
                        make_vertex(S1[ijack][_P ][mr_fw],S2[ijack][_LO][mr_bw],igam);
                    
                    // Pbw
                    jVert_LO_EM_P_S[Pbw][ijack][mr_fw][mr_bw][igam] +=
                        make_vertex(S1[ijack][_LO][mr_fw],S2[ijack][_P ][mr_bw],igam);
                    
                    // Sfw
                    jVert_LO_EM_P_S[Sfw][ijack][mr_fw][mr_bw][igam] +=
                    make_vertex(S1[ijack][_S ][mr_fw],S2[ijack][_LO][mr_bw],igam);
                    
                    // Sbw
                    jVert_LO_EM_P_S[Sbw][ijack][mr_fw][mr_bw][igam] +=
                    make_vertex(S1[ijack][_LO][mr_fw],S2[ijack][_S ][mr_bw],igam);

                }
    
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