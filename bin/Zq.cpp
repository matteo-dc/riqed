#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <omp.h>

//vprop_t GAMMA=make_gamma();

vvvd_t compute_jZq(vvvprop_t &jS_inv,const int imom)
{
    double V=size[0]*size[1]*size[2]*size[3];
    
    prop_t p_slash(prop_t::Zero());
    
    vvprop_t jS_0_inv = jS_inv[LO];
    vvprop_t jS_em_inv = jS_inv[EM];
    
    // vvdcompl_t jZq(vdcompl_t(0.0,nmr),njacks);
    vvd_t jZq_0_real(vd_t(0.0,nmr),njacks);
    vvd_t jZq_em_real(vd_t(0.0,nmr),njacks);
    
    vvvd_t jZq_LO_and_EM(vvd_t(vd_t(0.0,nmr),njacks),2);

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
        {
            jZq_0_real[ijack][mr]=-(I*(p_slash*jS_0_inv[ijack][mr]).trace()).real()/p2_tilde[imom]/12./V;
            jZq_em_real[ijack][mr]=-(I*(p_slash*jS_em_inv[ijack][mr]).trace()).real()/p2_tilde[imom]/12./V;
        }
    }
    if(UseSigma1==1) // using Sigma1
    {
        vvprop_t A_0(vprop_t(prop_t::Zero(),nmr),njacks);
        vvprop_t A_em(vprop_t(prop_t::Zero(),nmr),njacks);
        
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<nmr;mr++)
            {
                for(int igam=1;igam<5;igam++)
                    if(p_tilde[imom][igam%4]!=0.)
                    {
                        A_0[ijack][mr]+=GAMMA[igam]*jS_0_inv[ijack][mr]/p_tilde[imom][igam%4];
                        A_em[ijack][mr]+=GAMMA[igam]*jS_em_inv[ijack][mr]/p_tilde[imom][igam%4];
                    }
                A_0[ijack][mr]/=(double)count;
                A_em[ijack][mr]/=(double)count;
                
                jZq_0_real[ijack][mr]=-(I*A_0[ijack][mr].trace()).real()/12./V;
                jZq_em_real[ijack][mr]=-(I*A_em[ijack][mr].trace()).real()/12./V;
            }
    }
    
    jZq_LO_and_EM[LO]=jZq_0_real;
    jZq_LO_and_EM[EM]=jZq_em_real;
    
//    if(imom==10)
//        for(int ijack=0;ijack<njacks;ijack++)
//            printf("%lf\n",jZq_LO_and_EM[LO][ijack][0]);
    
    return jZq_LO_and_EM;
}