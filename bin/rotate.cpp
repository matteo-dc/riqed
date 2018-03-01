#include "Dirac.hpp"
#include "global.hpp"
#include "aliases.hpp"
#include <omp.h>
#include <valarray>
#include <complex>

// Rotation from Physical basis to Twisted basis

// quark
vvvprop_t rotate(vvvprop_t &S)
{
    int _njacks=S.size();
    int _ntypes=S[0].size();
    int _nmr=S[0][0].size();
    
    vvvprop_t S_rotated(vvprop_t(vprop_t(prop_t::Zero(),_nmr),_ntypes),_njacks);
    
    dcompl I(0.0,1.0);
    
#pragma omp parallel for collapse(2)
    for(int t=0;t<_ntypes;t++)
        for(int ijack=0;ijack<_njacks;ijack++)
            for(int mr=0;mr<_nmr;mr++)
            {
                int r = mr%nr;
                
                prop_t rot=(GAMMA[0]+I*(double)(2*r-1)*GAMMA[5])/sqrt(2);
                
                S_rotated[ijack][t][mr] = rot*S[ijack][t][mr]*rot;
            }
    
    return S_rotated;
}

// lepton
vvprop_t rotate(vvprop_t &S)
{
    int _njacks=S.size();
    int _ntypes=S[0].size();
    
    vvprop_t S_rotated(vprop_t(prop_t::Zero(),_ntypes),_njacks);
    
    dcompl I(0.0,1.0);
    
#pragma omp parallel for collapse(2)
    for(int t=0;t<_ntypes;t++)
        for(int ijack=0;ijack<_njacks;ijack++)
            {
                int r = 0;
                
                prop_t rot=(GAMMA[0]+I*(double)(2*r-1)*GAMMA[5])/sqrt(2);
                
                S_rotated[ijack][t] = rot*S[ijack][t]*rot;
            }
    
    return S_rotated;
}