#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"


void oper_t::build_prop(const vvvprop_t &prop, jprop_t &jS_LO,jprop_t &jS_PH,jprop_t &jS_P, jprop_t &jS_S)
{
    //#pragma omp parallel for collapse(3)
    for(int m=0;m<nm;m++)
        for(int r=0;r<nr;r++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                int mr = r + nr*m;
                
                jS_LO[ijack][mr]  += prop[ijack][0][mr];  // Leading order
                jS_PH[ijack][mr]  += prop[ijack][2][mr] + prop[ijack][3][mr];  // self energy + tadpole
                jS_P[ijack][mr]   += prop[ijack][4][mr]; // P insertion
                jS_S[ijack][mr]   += prop[ijack][5][mr]; // S insertion
                
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