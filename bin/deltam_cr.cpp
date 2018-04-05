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
    vvvvd_t jV0P5_LO(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_LL(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_0M(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_M0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_0T(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_T0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_QED(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_0P(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_P0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_P(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_0S(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_S0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jV0P5_S(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);

    //define jackknife P5P5 correlators
    vvvvd_t jP5P5_LO(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_LL(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jP5P5_0M(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jP5P5_M0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jP5P5_0T(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jP5P5_T0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_QED(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_0P(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jP5P5_P0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_P(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_0S(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jP5P5_S0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    vvvvd_t jP5P5_S(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);

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
    
    vvvvvd_t v_deltamcr(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
    vvvvvd_t v_deltamu(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm),nm);
    
#pragma omp parallel for collapse (3)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int r_fw=0;r_fw<nr;r_fw++)
            {
                int r_bw = r_fw; // same r
                
                int mr_fw = r_fw+nr*m_fw;
                int mr_bw = r_bw+nr*m_bw;
                
                //load V0P5 correlator
                jV0P5_LO[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"0","V0P5","IM","UNK",conf_id,path_to_ens);
                
                //load V0P5 corrections
                jV0P5_LL[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"F",mr_fw,"F","V0P5","IM","UNK",conf_id,path_to_ens);
                jV0P5_0M[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"FF","V0P5","IM","UNK",conf_id,path_to_ens);
                jV0P5_0T[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"T","V0P5","IM","UNK",conf_id,path_to_ens);
                jV0P5_0P[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"P","V0P5","RE","UNK",conf_id,path_to_ens);
                jV0P5_0S[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"S","V0P5","IM","UNK",conf_id,path_to_ens);
                jV0P5_M0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"FF",mr_fw,"0","V0P5","IM","UNK",conf_id,path_to_ens);
                jV0P5_T0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"T",mr_fw,"0","V0P5","IM","UNK",conf_id,path_to_ens);
                jV0P5_P0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"P",mr_fw,"0","V0P5","RE","UNK",conf_id,path_to_ens);
                jV0P5_S0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"S",mr_fw,"0","V0P5","IM","UNK",conf_id,path_to_ens);
                
                //collect V0P5 corrections
                jV0P5_QED = jV0P5_LL + jV0P5_0M + jV0P5_0T + jV0P5_M0 + jV0P5_T0;
                jV0P5_P = jV0P5_0P - jV0P5_P0;
                jV0P5_S = jV0P5_0S + jV0P5_S0;

                //load P5P5 correlator
                jP5P5_LO[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"0","P5P5","RE","UNK",conf_id,path_to_ens);

                //load P5P5 corrections
                jP5P5_LL[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"F",mr_fw,"F","P5P5","RE","UNK",conf_id,path_to_ens);
                jP5P5_0M[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"FF","P5P5","RE","UNK",conf_id,path_to_ens);
                jP5P5_0T[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"T","P5P5","RE","UNK",conf_id,path_to_ens);
                jP5P5_0P[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"P","P5P5","IM","UNK",conf_id,path_to_ens);
                jP5P5_0S[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"S","P5P5","RE","UNK",conf_id,path_to_ens);
                jP5P5_M0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"FF",mr_fw,"0","P5P5","RE","UNK",conf_id,path_to_ens);
                jP5P5_T0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"T",mr_fw,"0","P5P5","RE","UNK",conf_id,path_to_ens);
                jP5P5_P0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"P",mr_fw,"0","P5P5","IM","UNK",conf_id,path_to_ens);
                jP5P5_S0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"S",mr_fw,"0","P5P5","RE","UNK",conf_id,path_to_ens);
                
                //collect P5P5 corrections
                jP5P5_QED = jP5P5_LL + jP5P5_0M + jP5P5_0T + jP5P5_M0 + jP5P5_T0;
                jP5P5_P = jP5P5_0P - jP5P5_P0;
                jP5P5_S = jP5P5_0S + jP5P5_S0;
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
    vvvvd_t mean_value_mcr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm), sqr_mean_value_mcr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm), error_mcr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),nr),nm),nm);
    
    double coeff_P[2]={1.0,-1.0};
    double coeff_S=-1.0;
    
#warning taking only the insertions on the forward propagator
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int t=0;t<T/2/*+1*/;t++)
                for(int r=0;r<nr;r++)
                {
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
                        int mr_fw = r+nr*m_fw;
                        int mr_bw = r+nr*m_bw;
                        
                        // Solving with Kramer:
                        //   delta(V0P5):  a + b*deltamu + c*deltamcr + (correction to denominator) = 0
                        //   delta(slope[P5P5]):  d + e*deltamu + f*deltamcr = 0
                        
                        double a = symmetric_derivative(jV0P5_QED[mr_fw][mr_bw][ijack])[t]/jP5P5_LO[mr_fw][mr_bw][ijack][t] -
                                   symmetric_derivative(jV0P5_LO[mr_fw][mr_bw][ijack])[t]*jP5P5_QED[mr_fw][mr_bw][ijack][t]/jP5P5_LO[mr_fw][mr_bw][ijack][t]/jP5P5_LO[mr_fw][mr_bw][ijack][t];
                        double b = coeff_S*(symmetric_derivative(jV0P5_S[mr_fw][mr_bw][ijack])[t]/jP5P5_LO[mr_fw][mr_bw][ijack][t] -
                                            symmetric_derivative(jV0P5_LO[mr_fw][mr_bw][ijack])[t]*jP5P5_S[mr_fw][mr_bw][ijack][t]/jP5P5_LO[mr_fw][mr_bw][ijack][t]/jP5P5_LO[mr_fw][mr_bw][ijack][t]);
                        double c = coeff_P[r]*(symmetric_derivative(jV0P5_P[mr_fw][mr_bw][ijack])[t]/jP5P5_LO[mr_fw][mr_bw][ijack][t] - symmetric_derivative(jV0P5_LO[mr_fw][mr_bw][ijack])[t]*jP5P5_P[mr_fw][mr_bw][ijack][t]/jP5P5_LO[mr_fw][mr_bw][ijack][t]/jP5P5_LO[mr_fw][mr_bw][ijack][t]);
                        
                        double d = effective_slope(jP5P5_QED[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],eff_mass_time[mr_fw][mr_bw][ijack],T/2)[t];
                        double e = coeff_S*effective_slope(jP5P5_S[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],eff_mass_time[mr_fw][mr_bw][ijack],T/2)[t];
                        double f = coeff_P[r]*effective_slope(jP5P5_P[mr_fw][mr_bw][ijack]/jP5P5_LO[mr_fw][mr_bw][ijack],eff_mass_time[mr_fw][mr_bw][ijack],T/2)[t];
                        
                        double den = b*f-c*e;
                        double deltamu  = (-a*f+c*d)/den;
                        double deltamcr = (-b*d+a*e)/den;
                        
                        v_deltamu [m_fw][m_bw][r][ijack][t] = deltamu;
                        v_deltamcr[m_fw][m_bw][r][ijack][t] = deltamcr;
                        
//                        printf("r: %d m_fw: %d m_bw: %d ijack: %d t: %d deltamu: %lg deltamcr: %lg\n",r,m_fw,m_bw,ijack,t,deltamu,deltamcr);
                        
                        mean_value_mu[m_fw][m_bw][r][t]  += deltamu/njacks;
                        mean_value_mcr[m_fw][m_bw][r][t] += deltamcr/njacks;
                        
                        sqr_mean_value_mu[m_fw][m_bw][r][t]  += deltamu*deltamu/njacks;
                        sqr_mean_value_mcr[m_fw][m_bw][r][t] += deltamcr*deltamcr/njacks;
                    }
                    
                    error_mu[m_fw][m_bw][r][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mu[m_fw][m_bw][r][t]-mean_value_mu[m_fw][m_bw][r][t]*mean_value_mu[m_fw][m_bw][r][t]));
                    error_mcr[m_fw][m_bw][r][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mcr[m_fw][m_bw][r][t]-mean_value_mcr[m_fw][m_bw][r][t]*mean_value_mcr[m_fw][m_bw][r][t]));
                }
    
    vvd_t coord(vd_t(0.0,T/2+1),1);
    for(int j=0; j<T/2+1; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvvd_t jdeltamcr(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nr),nm),nm);
    vvvvvd_t jdeltamu(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nr),nm),nm);
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int r=0;r<nr;r++)
            {
                jdeltamcr[m_fw][m_bw][r] = polyfit(coord, 1, error_mcr[m_fw][m_bw][r], v_deltamcr[m_fw][m_bw][r], delta_tmin, delta_tmax);
                jdeltamu[m_fw][m_bw][r]  = polyfit(coord, 1, error_mu[m_fw][m_bw][r], v_deltamu[m_fw][m_bw][r], delta_tmin, delta_tmax);
            }
    
    vvvvd_t deltam_cr(vvvd_t(vvd_t(vd_t(0.0,nr),nm),nm),njacks);
    vvvvd_t deltamu(vvvd_t(vvd_t(vd_t(0.0,nr),nm),nm),njacks);
    
#pragma omp parallel for collapse(4)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int r=0;r<nr;r++)
                {
                    deltam_cr[ijack][m_fw][m_bw][r]=jdeltamcr[m_fw][m_bw][r][ijack][0];
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
                        outfile_mc.write((char*) &deltam_cr[ijack][m_fw][m_bw][r],sizeof(double));
                        outfile_mu.write((char*) &deltamu[ijack][m_fw][m_bw][r],sizeof(double));
                    }
        
        outfile_mu.close();
        outfile_mc.close();
    }
    else cerr<< "Unable to create the output file \"deltam_cr_array\" and \"deltamu_array\" "<<endl;
    
}
