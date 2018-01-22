#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <stdio.h>
#include <omp.h>

////amputate external legs
//jvert_t amputate(jprop_t &jprop1_inv,  jvert_t &jVert,  jprop_t &jprop2_inv)
//{
//    jvert_t jLambda(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks);
//
//#pragma omp parallel for collapse(4)
//    for(int ijack=0;ijack<njacks;ijack++)
//        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
//            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
//                for(int igam=0;igam<16;igam++)
//                {
//                    jLambda[ijack][mr_fw][mr_bw][igam]=jprop1_inv[ijack][mr_fw]*jVert[ijack][mr_fw][mr_bw][igam]*GAMMA[5]*(jprop2_inv[ijack][mr_bw]).adjoint()*GAMMA[5];
//                }
//    
//    return jLambda;
//}
//
////project the amputated green function
//jproj_t project(jvert_t &jLambda)
//{
//    //L_proj has 5 components: S(0), V(1), P(2), A(3), T(4)
//    jvert_t L_proj(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),5),nmr),nmr),njacks);
//    jproj_t jG(vvvd_t(vvd_t(vd_t(0.0,5),nmr),nmr),njacks);
//    
//#pragma omp parallel for collapse(3)
//    for(int ijack=0;ijack<njacks;ijack++)
//        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
//            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
//            {
//                L_proj[ijack][mr_fw][mr_bw][0]=jLambda[ijack][mr_fw][mr_bw][0]*P[0];
//                for(int igam=1;igam<5;igam++)
//                    L_proj[ijack][mr_fw][mr_bw][1]+=jLambda[ijack][mr_fw][mr_bw][igam]*P[igam];
//                L_proj[ijack][mr_fw][mr_bw][2]=jLambda[ijack][mr_fw][mr_bw][5]*P[5];
//                for(int igam=6;igam<10;igam++)
//                    L_proj[ijack][mr_fw][mr_bw][3]+=jLambda[ijack][mr_fw][mr_bw][igam]*P[igam];
//                for(int igam=10;igam<16;igam++)
//                    L_proj[ijack][mr_fw][mr_bw][4]+=jLambda[ijack][mr_fw][mr_bw][igam]*P[igam];
//                
//                for(int j=0;j<5;j++)
//                    jG[ijack][mr_fw][mr_bw][j]=L_proj[ijack][mr_fw][mr_bw][j].trace().real()/12.0;
//            }
//    
//    return jG;
//}

//project the amputated green function
vvvvvd_t compute_pr_bil( vvvprop_t &jprop1_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jprop2_inv)
{
   
    int i1[4]={LO,LO,EM,LO};
    int iv[4]={LO,EM,LO,LO};
    int i2[4]={LO,LO,LO,EM};
    
//    jvert_t jLambda=amputate(jS1_inv, jVert, jS2_inv);
//    jproj_t jG=project(jLambda);
    
    valarray<jvert_t> jLambda(vvvvprop_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),nmr),nmr),njacks),4);
//    jvert_t L_proj(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),5),nmr),nmr),njacks);  // L_proj has 5 components: S,V,P,A,T

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
                        
                        if(igam==0) /*L_proj[ijack][mr_fw][mr_bw][0]*/jG[k][0][ijack][mr_fw][mr_bw]=(jLambda[k][ijack][mr_fw][mr_bw][0]*P[0]).trace().real()/12.0;
                        if(igam>0 and igam<5) jG[k][1][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*P[igam]).trace().real()/12.0;
                        if(igam==5) jG[k][2][ijack][mr_fw][mr_bw]=(jLambda[k][ijack][mr_fw][mr_bw][5]*P[5]).trace().real()/12.0;
                        if(igam>5 and igam<10) jG[k][3][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*P[igam]).trace().real()/12.0;
                        if(igam>=10 and igam<16) jG[k][4][ijack][mr_fw][mr_bw]+=(jLambda[k][ijack][mr_fw][mr_bw][igam]*P[igam]).trace().real()/12.0;
                        
                    }
    //                for(int j=0;j<5;j++)
    //                    jG[ijack][mr_fw][mr_bw][j]=L_proj[ijack][mr_fw][mr_bw][j].trace().real()/12.0;
    //            }
    
    
    jG_LO_and_EM[LO] = jG[0];
    jG_LO_and_EM[EM] = -jG[1]+jG[2]+jG[3];  // jG_em = -jG_1+jG_a+jG_b;
    
    return jG_LO_and_EM;
}


