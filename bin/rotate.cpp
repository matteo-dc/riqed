#include "Dirac.hpp"
#include "global.hpp"
#include "aliases.hpp"
#include <omp.h>
#include <valarray>
#include <complex>

// Rotation from Physical basis to Twisted basis
vvvprop_t rotate(vvvprop_t &S)
{
    vvvprop_t S_rotated(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
    
    dcompl I(0.0,1.0);
    
#pragma omp parallel for collapse(3)
    for(int m=0;m<nm;m++)
        for(int t=0;t<ntypes;t++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int r=0;r<nr;r++)
                    {
                        prop_t rot=(GAMMA[0]+I*(double)(2*r-1)*GAMMA[5])/sqrt(2);
                        
                        int mr = r + nr*m;
                        
                        S_rotated[ijack][t][mr] = rot*S[ijack][t][mr]*rot;
                    }
    
    return S_rotated;
}