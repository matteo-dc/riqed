#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <stdio.h>
#include <omp.h>


//project the amputated green function
vvvvvd_t compute_pr_bil( vvvprop_t &jprop1_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jprop2_inv)
{
   
    int i1[4]={LO,LO,EM,LO};
    int iv[4]={LO,EM,LO,LO};
    int i2[4]={LO,LO,LO,EM};
    
    valarray<jvert_t> jLambda(vvvvprop_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks),4);

    valarray<jproj_t> jG(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil),4);
    valarray<jproj_t> jG_LO_and_EM(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil),2);
    
#pragma omp parallel for collapse(5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int igam=0;igam<16;igam++)
                    for(int k=0;k<4;k++)
                    {
                        jLambda[k][ijack][mr_fw][mr_bw][igam]=jprop1_inv[i1[k]][ijack][mr_fw]*jVert[iv[k]][ijack][mr_fw][mr_bw][igam]*GAMMA[5]*(jprop2_inv[i2[k]][ijack][mr_bw]).adjoint()*GAMMA[5];
                        
                        if(igam==0) jG[k][0][ijack][mr_fw][mr_bw]=(jLambda[k][ijack][mr_fw][mr_bw][0]*P[0]).trace().real()/12.0;
                        if(igam>0 and igam<5) jG[k][1][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*P[igam]).trace().real()/12.0;
                        if(igam==5) jG[k][2][ijack][mr_fw][mr_bw]=(jLambda[k][ijack][mr_fw][mr_bw][5]*P[5]).trace().real()/12.0;
                        if(igam>5 and igam<10) jG[k][3][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*P[igam]).trace().real()/12.0;
                        if(igam>=10 and igam<16) jG[k][4][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*P[igam]).trace().real()/12.0;
                        
                    }
    
    jG_LO_and_EM[LO] = jG[0];
    jG_LO_and_EM[EM] = -jG[1]+jG[2]+jG[3];  // jG_em = -jG_1+jG_a+jG_b;
    
    return jG_LO_and_EM;
}


