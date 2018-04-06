#include "global.hpp"
#include "aliases.hpp"
#include "contractions.hpp"
#include "jack.hpp"
#include "fit.hpp"
#include <iostream>
#include <omp.h>
#include "operations.hpp"
#include "deltam_cr.hpp"
#include "tools.hpp"


//compute delta m_cr
void oper_t::compute_deltam()
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    //define jackknife V0P5 correlators
    vvvvd_t jV0P5_LO(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_LL(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jV0P5_0M(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jV0P5_M0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jV0P5_0T(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jV0P5_T0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_QED(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_0P(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jV0P5_P0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_P(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_0S(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jV0P5_S0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_S(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);

    //define jackknife P5P5 correlators
    vvvvd_t jP5P5_LO(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_LL(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jP5P5_0M(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jP5P5_M0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jP5P5_0T(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jP5P5_T0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_QED(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_0P(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jP5P5_P0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_P(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_0S(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    vvvvd_t jP5P5_S0(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_S(vvvd_t(vvd_t(vd_t(T),njacks),nmr),nmr);

//    //define jackknife V0P5 correlators with a unique r dependence
//    vvvvvd_t jV0P5_LL_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jV0P5_0M_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jV0P5_M0_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jV0P5_0T_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jV0P5_T0_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jV0P5_0P_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jV0P5_P0_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//
//    //define jackknife P5P5 correlators with a unique r dependence
//    vvvvvd_t jP5P5_LL_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jP5P5_0M_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jP5P5_M0_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jP5P5_0T_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jP5P5_T0_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jP5P5_0P_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t jP5P5_P0_ave(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);

    //define deltam_cr
//    vvvvvd_t num_deltam_cr_corr(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
//    vvvvvd_t den_deltam_cr_corr(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
    
    vvvvvd_t v_deltamc(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
    vvvvvd_t v_deltamu(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
    
    // load correlators
#pragma omp parallel for collapse (2)
    for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
        {
            int r_fw = mr_fw%nr;
            int m_fw = (mr_fw-r_fw)/nr;
            int r_bw = mr_bw%nr;
            int m_bw = (mr_bw-r_bw)/nr;
            
            //load V0P5 correlator
            jV0P5_LO[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            
            //load V0P5 corrections
            jV0P5_LL[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_F ,_F ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0M[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_FF,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0T[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_T ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0P[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_P ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0S[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_S ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_M0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_FF,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_T0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_T ,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_P0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_P ,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_S0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_S ,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            
            //load P5P5 correlator
            jP5P5_LO[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            
            //load P5P5 corrections
            jP5P5_LL[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_F ,_F ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0M[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_FF,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0T[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_T ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0P[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_P ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0S[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_LO,_S ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_M0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_FF,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_T0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_T ,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_P0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_P ,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_S0[mr_fw][mr_bw]=get_contraction("",out_hadr,m_fw,m_bw,r_fw,r_bw,_S ,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
        }
    
#pragma omp parallel for collapse (3)
    for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
            for (int ijack=0; ijack<njacks; ijack++)
            {
                //collect V0P5 corrections
                jV0P5_QED[mr_fw][mr_bw][ijack] = jV0P5_LL[mr_fw][mr_bw][ijack] +
                                                 jV0P5_0M[mr_fw][mr_bw][ijack] +
                                                 jV0P5_0T[mr_fw][mr_bw][ijack] +
                                                 jV0P5_M0[mr_fw][mr_bw][ijack] +
                                                 jV0P5_T0[mr_fw][mr_bw][ijack];
                
                jV0P5_P[mr_fw][mr_bw][ijack]   = jV0P5_0P[mr_fw][mr_bw][ijack] +
                                                 jV0P5_P0[mr_fw][mr_bw][ijack];
                
                jV0P5_S[mr_fw][mr_bw][ijack]   = jV0P5_0S[mr_fw][mr_bw][ijack] +
                                                 jV0P5_S0[mr_fw][mr_bw][ijack];
                
                //collect P5P5 corrections
                jP5P5_QED[mr_fw][mr_bw][ijack] = jP5P5_LL[mr_fw][mr_bw][ijack] +
                                                 jP5P5_0M[mr_fw][mr_bw][ijack] +
                                                 jP5P5_0T[mr_fw][mr_bw][ijack] +
                                                 jP5P5_M0[mr_fw][mr_bw][ijack] +
                                                 jP5P5_T0[mr_fw][mr_bw][ijack];
                
                jP5P5_P[mr_fw][mr_bw][ijack]   = jP5P5_0P[mr_fw][mr_bw][ijack] +
                                                 jP5P5_P0[mr_fw][mr_bw][ijack];
                
                jP5P5_S[mr_fw][mr_bw][ijack]   = jP5P5_0S[mr_fw][mr_bw][ijack] +
                                                 jP5P5_S0[mr_fw][mr_bw][ijack];
            }
    
//    //average over r
//#pragma omp parallel for collapse(4)
//    for(int m_fw=0;m_fw<nm;m_fw++)
//        for(int m_bw=0;m_bw<nm;m_bw++)
//            for (int ijack=0; ijack<njacks; ijack++)
//                for(int t=0;t<T/2+1;t++)
//                    for(int r=0;r<nr;r++)
//                    {
//                        int r1=r;
//                        int r2=(r+1)%2;
//
//                        int cr_even = r2 + r1;
//                        int cr_odd = r2 - r1;
//
//                        jV0P5_LL_ave[m_fw][m_bw][ijack][t] += jV0P5_LL[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_odd/nr;
//                        jV0P5_0M_ave[m_fw][m_bw][ijack][t] += jV0P5_0M[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_odd/nr;
//                        jV0P5_M0_ave[m_fw][m_bw][ijack][t] += jV0P5_M0[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_odd/nr;
//                        jV0P5_0T_ave[m_fw][m_bw][ijack][t] += jV0P5_0T[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_odd/nr;
//                        jV0P5_T0_ave[m_fw][m_bw][ijack][t] += jV0P5_T0[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_odd/nr;
//                        jV0P5_0P_ave[m_fw][m_bw][ijack][t] += jV0P5_0P[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_even/nr;
//                        jV0P5_P0_ave[m_fw][m_bw][ijack][t] += jV0P5_P0[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_even/nr;
//                    }
    
//    // same r but without average
//#pragma omp parallel for collapse(5)
//    for(int m_fw=0;m_fw<nm;m_fw++)
//        for(int m_bw=0;m_bw<nm;m_bw++)
//            for (int ijack=0; ijack<njacks; ijack++)
//                for(int t=0;t<T/2+1;t++)
//                    for(int r=0;r<nr;r++)
//                    {
//                        int r1=r;
//                        int r2=(r+1)%2;
//                        
//                        jV0P5_LL_ave[m_fw][m_bw][r][ijack][t] = jV0P5_LL[r1+nr*m_fw][r1+nr*m_bw][ijack][t];
//                        jV0P5_0M_ave[m_fw][m_bw][r][ijack][t] = jV0P5_0M[r1+nr*m_fw][r1+nr*m_bw][ijack][t];
//                        jV0P5_M0_ave[m_fw][m_bw][r][ijack][t] = jV0P5_M0[r1+nr*m_fw][r1+nr*m_bw][ijack][t];
//                        jV0P5_0T_ave[m_fw][m_bw][r][ijack][t] = jV0P5_0T[r1+nr*m_fw][r1+nr*m_bw][ijack][t];
//                        jV0P5_T0_ave[m_fw][m_bw][r][ijack][t] = jV0P5_T0[r1+nr*m_fw][r1+nr*m_bw][ijack][t];
//                        jV0P5_0P_ave[m_fw][m_bw][r][ijack][t] = jV0P5_0P[r1+nr*m_fw][r1+nr*m_bw][ijack][t];
//                        jV0P5_P0_ave[m_fw][m_bw][r][ijack][t] = jV0P5_P0[r1+nr*m_fw][r1+nr*m_bw][ijack][t];
//                    }
    
   
//    // taking only opposite r (opposite r's become same r's due to the dagger):
//    //   in Nf=4 analysis we are not at maximal twist and the leading order m_cr wouldn't be zero otherwise.
//#pragma omp parallel for collapse(4)
//    for(int m_fw=0;m_fw<nm;m_fw++)
//        for(int m_bw=0;m_bw<nm;m_bw++)
//            for (int ijack=0; ijack<njacks; ijack++)
//                for(int t=0;t<T/2+1;t++)
//                    for(int r=0; r<1; r++)
//                    {
//                        int r1=r;
//                        int r2=(r+1)%2;
//                        
//                        jV0P5_LL_ave[m_fw][m_bw][ijack][t] = jV0P5_LL[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
//                        jV0P5_0M_ave[m_fw][m_bw][ijack][t] = jV0P5_0M[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
//                        jV0P5_M0_ave[m_fw][m_bw][ijack][t] = jV0P5_M0[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
//                        jV0P5_0T_ave[m_fw][m_bw][ijack][t] = jV0P5_0T[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
//                        jV0P5_T0_ave[m_fw][m_bw][ijack][t] = jV0P5_T0[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
//                        jV0P5_0P_ave[m_fw][m_bw][ijack][t] = jV0P5_0P[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
//                        jV0P5_P0_ave[m_fw][m_bw][ijack][t] = jV0P5_P0[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
//                    }
    
    vvvvd_t mean_value_mu(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm), sqr_mean_value_mu(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm), error_mu(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm);
    vvvvd_t mean_value_mc(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm), sqr_mean_value_mc(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm), error_mc(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm);
    
//    //DEBUG
//    for(int m_fw=0;m_fw<1;m_fw++)
//        for(int m_bw=0;m_bw<1;m_bw++)
//            for(int r=0;r<nr;r++)
//                for(int ijack=0;ijack<njacks;ijack++)
//                    for(int t=0;t<T/2/*+1*/;t++)
//                    {
//                        int mr_fw = r+nr*m_fw;
//                        int mr_bw = r+nr*m_bw;
//                        
//                        cout<<"mfw: "<<m_fw<<" mbw: "<<m_bw<<" r: "<<r<<" ijack: "<<ijack<<" t: "<<t<<" slope: ";
//                        cout<<effective_slope(jP5P5_P[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],eff_mass_time[mr_fw][mr_bw][ijack],T/2)[t]<<endl;
//                    }
//    exit(1);
//    //////DEBUG
    
    vd_t A(T/2+1),B(T/2+1),C(T/2+1),D(T/2+1),E(T/2+1),F(T/2+1);
    
    // Solving with Kramer:
    //   delta(mPCAC):                  a + b*deltamu + c*deltamcr + (correction to denominator) = 0
    //   delta(slope[P5P5_ins/P5P5]):   d + e*deltamu + f*deltamcr = 0
    
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int r=0;r<nr;r++)
                {
                    int mr_fw = r+nr*m_fw;
                    int mr_bw = r+nr*m_bw;
                    
                    A = symmetrize(symmetric_derivative(jV0P5_QED[mr_fw][mr_bw][ijack])/jP5P5_LO[mr_fw][mr_bw][ijack] - symmetric_derivative(jV0P5_LO[mr_fw][mr_bw][ijack])*jP5P5_QED[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],0);
                    B = symmetrize(symmetric_derivative(jV0P5_S[mr_fw][mr_bw][ijack])/jP5P5_LO[mr_fw][mr_bw][ijack] - symmetric_derivative(jV0P5_LO[mr_fw][mr_bw][ijack])*jP5P5_S[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],0);
                    C = symmetrize(symmetric_derivative(jV0P5_P[mr_fw][mr_bw][ijack])/jP5P5_LO[mr_fw][mr_bw][ijack] - symmetric_derivative(jV0P5_LO[mr_fw][mr_bw][ijack])*jP5P5_P[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],0);
                    
                    D = effective_slope(symmetrize(jP5P5_QED[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],0),eff_mass_time[mr_fw][mr_bw][ijack],T/2);
                    E = effective_slope(symmetrize(jP5P5_S[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],0),eff_mass_time[mr_fw][mr_bw][ijack],T/2);
                    F = effective_slope(symmetrize(jP5P5_P[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],0),eff_mass_time[mr_fw][mr_bw][ijack],T/2);
                    
                    vd_t den = B*F-C*E;
                    vd_t deltamu = (-A*F+C*D)/den;
                    vd_t deltamc = (-B*D+A*E)/den;
                    
                    v_deltamu[m_fw][m_bw][r][ijack] = deltamu;
                    v_deltamc[m_fw][m_bw][r][ijack] = deltamc;
                }
    
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int t=0;t<T/2/*+1*/;t++)
                for(int r=0;r<nr;r++)
                {
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
                        mean_value_mu[m_fw][m_bw][r][t] += v_deltamu[m_fw][m_bw][r][ijack][t]/njacks;
                        mean_value_mc[m_fw][m_bw][r][t] += v_deltamc[m_fw][m_bw][r][ijack][t]/njacks;
                        
                        sqr_mean_value_mu[m_fw][m_bw][r][t] += v_deltamu[m_fw][m_bw][r][ijack][t]*v_deltamu[m_fw][m_bw][r][ijack][t]/njacks;
                        sqr_mean_value_mc[m_fw][m_bw][r][t] += v_deltamc[m_fw][m_bw][r][ijack][t]*v_deltamc[m_fw][m_bw][r][ijack][t]/njacks;
                    }
                    
                    error_mu[m_fw][m_bw][r][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mu[m_fw][m_bw][r][t]-mean_value_mu[m_fw][m_bw][r][t]*mean_value_mu[m_fw][m_bw][r][t]));
                    error_mc[m_fw][m_bw][r][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mc[m_fw][m_bw][r][t]-mean_value_mc[m_fw][m_bw][r][t]*mean_value_mc[m_fw][m_bw][r][t]));
                }
    
    vvd_t coord(vd_t(0.0,T/2+1),1);
    for(int j=0; j<T/2+1; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvvd_t jdeltamc(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nr),nm),nm);
    vvvvvd_t jdeltamu(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nr),nm),nm);
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int r=0;r<nr;r++)
            {
                jdeltamc[m_fw][m_bw][r] = polyfit(coord, 1, error_mc[m_fw][m_bw][r], v_deltamc[m_fw][m_bw][r], delta_tmin, delta_tmax);
                jdeltamu[m_fw][m_bw][r] = polyfit(coord, 1, error_mu[m_fw][m_bw][r], v_deltamu[m_fw][m_bw][r], delta_tmin, delta_tmax);
            }
    
    vvvvd_t deltamc(vvvd_t(vvd_t(vd_t(0.0,nr),nm),nm),njacks);
    vvvvd_t deltamu(vvvd_t(vvd_t(vd_t(0.0,nr),nm),nm),njacks);
    
#pragma omp parallel for collapse(4)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int r=0;r<nr;r++)
                {
                    deltamc[ijack][m_fw][m_bw][r]=jdeltamc[m_fw][m_bw][r][ijack][0];
                    deltamu[ijack][m_fw][m_bw][r]=jdeltamu[m_fw][m_bw][r][ijack][0];
                }
    
    ofstream outfile_mu, outfile_mc;
    outfile_mc.open(path_to_ens+"deltam_cr_array", ios::out | ios::binary);
    outfile_mu.open(path_to_ens+"deltamu_array", ios::out | ios::binary);
    
    if (outfile_mc.is_open() and outfile_mu.is_open())
    {
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int r=0;r<nr;r++)
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
                        outfile_mc.write((char*) &deltamc[ijack][m_fw][m_bw][r],sizeof(double));
                        outfile_mu.write((char*) &deltamu[ijack][m_fw][m_bw][r],sizeof(double));
                    }
        
        outfile_mu.close();
        outfile_mc.close();
    }
    else cerr<< "Unable to create the output file \"deltam_cr_array\" and \"deltamu_array\" "<<endl;
    
}
