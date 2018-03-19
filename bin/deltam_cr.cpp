#include "global.hpp"
#include "aliases.hpp"
#include "contractions.hpp"
#include "jack.hpp"
#include "fit.hpp"
#include <iostream>
#include <omp.h>
#include "operations.hpp"


//compute delta m_cr
void oper_t::compute_deltam_cr()
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    //define jackknife V0P5 correlators
    vvvvd_t jV0P5_LL(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_0M(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_M0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_0T(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_T0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_0P(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    vvvvd_t jV0P5_P0(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    //define jackknife V0P5 correlators r independent
    vvvvd_t jV0P5_LL_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t jV0P5_0M_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t jV0P5_M0_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t jV0P5_0T_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t jV0P5_T0_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t jV0P5_0P_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t jV0P5_P0_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    
    
    //define deltam_cr
    vvvvd_t num_deltam_cr_corr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t den_deltam_cr_corr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    vvvvd_t deltam_cr_corr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
    
#pragma omp parallel for collapse (2)
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
        {
            //load corrections
            jV0P5_LL[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"F",mr_fw,"F","V0P5","IM","ODD",conf_id,path_to_ens);
            jV0P5_0M[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"FF","V0P5","IM","ODD",conf_id,path_to_ens);
            jV0P5_M0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"FF",mr_fw,"0","V0P5","IM","ODD",conf_id,path_to_ens);
            jV0P5_0T[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"T","V0P5","IM","ODD",conf_id,path_to_ens);
            jV0P5_T0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"T",mr_fw,"0","V0P5","IM","ODD",conf_id,path_to_ens);
            //load the derivative wrt counterterm
            jV0P5_0P[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"0",mr_fw,"P","V0P5","RE","ODD",conf_id,path_to_ens);
            jV0P5_P0[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_bw,"P",mr_fw,"0","V0P5","RE","ODD",conf_id,path_to_ens);
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
   
    // taking only opposite r (opposite r's become same r's due to the dagger):
    //   in Nf=4 analysis we are not at maximal twist and the leading order m_cr wouldn't be zero otherwise.
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for (int ijack=0; ijack<njacks; ijack++)
                for(int t=0;t<T/2+1;t++)
                    for(int r=0; r<1; r++)
                    {
                        int r1=r;
                        int r2=(r+1)%2;
                        
                        jV0P5_LL_ave[m_fw][m_bw][ijack][t] = jV0P5_LL[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
                        jV0P5_0M_ave[m_fw][m_bw][ijack][t] = jV0P5_0M[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
                        jV0P5_M0_ave[m_fw][m_bw][ijack][t] = jV0P5_M0[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
                        jV0P5_0T_ave[m_fw][m_bw][ijack][t] = jV0P5_0T[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
                        jV0P5_T0_ave[m_fw][m_bw][ijack][t] = jV0P5_T0[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
                        jV0P5_0P_ave[m_fw][m_bw][ijack][t] = jV0P5_0P[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
                        jV0P5_P0_ave[m_fw][m_bw][ijack][t] = jV0P5_P0[r1+nr*m_fw][r2+nr*m_bw][ijack][t];
                    }
    
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int t=0;t<T/2+1;t++)
                {
                    num_deltam_cr_corr[m_fw][m_bw][ijack][t] = jV0P5_LL_ave[m_fw][m_bw][ijack][t]+jV0P5_0M_ave[m_fw][m_bw][ijack][t]+jV0P5_M0_ave[m_fw][m_bw][ijack][t]+jV0P5_0T_ave[m_fw][m_bw][ijack][t]+jV0P5_T0_ave[m_fw][m_bw][ijack][t];
                    
                    den_deltam_cr_corr[m_fw][m_bw][ijack][t] = jV0P5_P0_ave[m_fw][m_bw][ijack][t]-jV0P5_0P_ave[m_fw][m_bw][ijack][t];
                    
                    deltam_cr_corr[m_fw][m_bw][ijack][t] = num_deltam_cr_corr[m_fw][m_bw][ijack][t]/den_deltam_cr_corr[m_fw][m_bw][ijack][t];
                
                    printf("m_fw: %d m_bw: %d ijack: %d t: %d delta: %lf \n",m_fw,m_bw,ijack,t,deltam_cr_corr[m_fw][m_bw][ijack][t]);
                }
    
    vvvd_t mean_value(vvd_t(vd_t(0.0,T/2+1),nm),nm), sqr_mean_value(vvd_t(vd_t(0.0,T/2+1),nm),nm), error(vvd_t(vd_t(0.0,T/2+1),nm),nm);
    
//#pragma omp parallel for collapse(3)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int t=0;t<T/2+1;t++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mean_value[m_fw][m_bw][t]+=deltam_cr_corr[m_fw][m_bw][ijack][t]/njacks;
                    sqr_mean_value[m_fw][m_bw][t]+=deltam_cr_corr[m_fw][m_bw][ijack][t]*deltam_cr_corr[m_fw][m_bw][ijack][t]/njacks;
                }
                error[m_fw][m_bw][t]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value[m_fw][m_bw][t]-mean_value[m_fw][m_bw][t]*mean_value[m_fw][m_bw][t]));
            }
    
    vvd_t coord(vd_t(0.0,T/2+1),1);
    for(int j=0; j<T/2+1; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvd_t jdeltam_cr(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nm),nm);
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            jdeltam_cr[m_fw][m_bw] = polyfit(coord, 1, error[m_fw][m_bw], deltam_cr_corr[m_fw][m_bw], delta_tmin, delta_tmax);
    
    vvvd_t deltam_cr(vvd_t(vd_t(0.0,nm),nm),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                deltam_cr[ijack][m_fw][m_bw]=jdeltam_cr[m_fw][m_bw][ijack][0];
    
    
    ofstream outfile;
    outfile.open(path_to_ens+"deltam_cr_array", ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m_fw=0;m_fw<nm;m_fw++)
                for(int m_bw=0;m_bw<nm;m_bw++)
                {
                    outfile.write((char*) &deltam_cr[ijack][m_fw][m_bw],sizeof(double));
               
//                    cout<<"ijack "<<ijack<<" m1 "<<m_fw<<" m2 "<<m_bw<<" delta: "<<deltam_cr/*_array*/[ijack][m_fw][m_bw]<<endl;
                }
        
        outfile.close();
    }
    else cerr<< "Unable to create the output file \"deltam_cr_array\" "<<endl;
    
}
