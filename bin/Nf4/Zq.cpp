#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"

//vprop_t GAMMA=make_gamma();

vvd_t compute_jZq(jprop_t jS_inv, int imom)
{
    double V=size[0]*size[1]*size[2]*size[3];
    
    prop_t p_slash(prop_t::Zero());
    
    // vvdcompl_t jZq(vdcompl_t(0.0,nmr),njacks);
    vvd_t jZq_real(vd_t(0.0,nmr),njacks);
    dcompl I(0.0,1.0);
    
    int count=0;
    
    for(int igam=1;igam<5;igam++)
    {
        p_slash+=GAMMA[igam]*p_tilde[imom][igam%4];
        
        if(p_tilde[imom][igam%4]!=0.)
            count++;
    }
    
    Np.push_back(count);
    
    /*  Note that: p_slash*p_slash=p2*GAMMA[0]  */
 
    if(UseSigma1==0) // using Zq
    {
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
            jZq_real[ijack][mr]=-(I*(p_slash*jS_inv[ijack][mr]).trace()).real()/p2_tilde[imom]/12./V;
    }
    if(UseSigma1==1) // using Sigma1
    {
        vvprop_t A(vprop_t(prop_t::Zero(),nmr),njacks);
        
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<nmr;mr++)
            {
                for(int igam=1;igam<5;igam++)
                    if(p_tilde[imom][igam%4]!=0.)
                    {
                        A[ijack][mr]+=GAMMA[igam]*jS_inv[ijack][mr]/p_tilde[imom][igam%4];
                    }
                A[ijack][mr]/=(double)count;
                jZq_real[ijack][mr]=-(I*A[ijack][mr].trace()).real()/12./V;
            }
    }
    
    return jZq_real;
    
}