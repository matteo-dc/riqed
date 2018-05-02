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

// compute deltam_cr and deltamu from propagators
void oper_t::compute_deltam_from_prop()
{
    //define deltam_cr
    vvvvd_t v_deltamc(vvvd_t(vvd_t(vd_t(0.0,_linmoms),njacks),nr),nm);
    vvvvd_t v_deltamu(vvvd_t(vvd_t(vd_t(0.0,_linmoms),njacks),nr),nm);
    
    vvvd_t  mean_value_mu(vvd_t(vd_t(0.0,_linmoms),nr),nm),
            sqr_mean_value_mu(vvd_t(vd_t(0.0,_linmoms),nr),nm),
            error_mu(vvd_t(vd_t(0.0,_linmoms),nr),nm);
    vvvd_t  mean_value_mc(vvd_t(vd_t(0.0,_linmoms),nr),nm),
            sqr_mean_value_mc(vvd_t(vd_t(0.0,_linmoms),nr),nm),
            error_mc(vvd_t(vd_t(0.0,_linmoms),nr),nm);
    
    // Solving with Kramer:
    //   B*(deltamu) + C*(deltamcr) - A = 0
    //   E*(deltamu) + F*(deltamcr) - D = 0
    
#pragma omp parallel for collapse(4)
    for(int imom=0;imom<_linmoms;imom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m=0;m<nm;m++)
                for(int r=0;r<nr;r++)
                {
                    int mr = r+nr*m;
                    
                    double A = sigma2_PH[imom][ijack][mr];
                    double B = sigma2_S[imom][ijack][mr];
                    double C = sigma2_P[imom][ijack][mr];
                    
                    double D = sigma3_PH[imom][ijack][mr];
                    double E = sigma3_S[imom][ijack][mr];
                    double F = sigma3_P[imom][ijack][mr];
                    
                    double den = B*F-C*E;
                    double deltamu = (-A*F+C*D)/den;
                    double deltamc = (-B*D+A*E)/den;
                    
                    v_deltamu[m][r][ijack][imom] = deltamu;
                    v_deltamc[m][r][ijack][imom] = deltamc;
                }

#pragma omp parallel for collapse(3)
    for(int imom=0;imom<_linmoms;imom++)
        for(int m=0;m<nm;m++)
            for(int r=0;r<nr;r++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mean_value_mu[m][r][imom] += v_deltamu[m][r][ijack][imom]/njacks;
                    mean_value_mc[m][r][imom] += v_deltamc[m][r][ijack][imom]/njacks;
                    
                    sqr_mean_value_mu[m][r][imom] += v_deltamu[m][r][ijack][imom]*v_deltamu[m][r][ijack][imom]/njacks;
                    sqr_mean_value_mc[m][r][imom] += v_deltamc[m][r][ijack][imom]*v_deltamc[m][r][ijack][imom]/njacks;
                }
                
                error_mu[m][r][imom] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mu[m][r][imom]-mean_value_mu[m][r][imom]*mean_value_mu[m][r][imom]));
                error_mc[m][r][imom] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mc[m][r][imom]-mean_value_mc[m][r][imom]*mean_value_mc[m][r][imom]));
            }
    
    int npar=3;
    vvd_t coord(vd_t(0.0,_linmoms),npar);
    for(int j=0; j<_linmoms; j++)
    {
        coord[0][j] = 1.0;
        coord[1][j] = p2[j];
        coord[2][j] = p2[j]*p2[j];
    }
    
    vvvd_t deltamc(vvd_t(vd_t(0.0,nr),nm),njacks);
    vvvd_t deltamu(vvd_t(vd_t(0.0,nr),nm),njacks);
    
    for(int m=0;m<nm;m++)
        for(int r=0;r<nr;r++)
        {
            vvd_t deltamc_tmp = polyfit(coord, npar, error_mc[m][r], v_deltamc[m][r], delta_tmin, delta_tmax);
            vvd_t deltamu_tmp = polyfit(coord, npar, error_mu[m][r], v_deltamu[m][r], delta_tmin, delta_tmax);
            
            for(int ijack=0;ijack<njacks;ijack++)
            {
                deltamc[ijack][m][r]=deltamc_tmp[ijack][0];
                deltamu[ijack][m][r]=deltamu_tmp[ijack][0];
            }
        }
    
    ofstream outfile_mu, outfile_mc;
    outfile_mc.open(path_to_ens+"deltam_cr_array", ios::out | ios::binary);
    outfile_mu.open(path_to_ens+"deltamu_array", ios::out | ios::binary);
    
    if (outfile_mc.is_open() and outfile_mu.is_open())
    {
        for(int m=0;m<nm;m++)
            for(int r=0;r<nr;r++)
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    outfile_mc.write((char*) &deltamc[ijack][m][r],sizeof(double));
                    outfile_mu.write((char*) &deltamu[ijack][m][r],sizeof(double));
                }
        
        outfile_mu.close();
        outfile_mc.close();
    }
    else cerr<< "Unable to create the output file \"deltam_cr_array\" and \"deltamu_array\" "<<endl;
    
}


// compute deltam_cr and deltamu from correlators
void oper_t::compute_deltam()
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    //define jackknife V0P5 correlators
    vvvd_t jV0P5_LO(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jV0P5_LL(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jV0P5_0M(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jV0P5_M0(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jV0P5_0T(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jV0P5_T0(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jV0P5_QED(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jV0P5_0P(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jV0P5_P0(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jV0P5_P(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jV0P5_0S(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jV0P5_S0(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jV0P5_S(vvd_t(vd_t(T),njacks),nmr);

    //define jackknife P5P5 correlators
    vvvd_t jP5P5_LO(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jP5P5_LL(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jP5P5_0M(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jP5P5_M0(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jP5P5_0T(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jP5P5_T0(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jP5P5_QED(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jP5P5_0P(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jP5P5_P0(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jP5P5_P(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jP5P5_0S(vvd_t(vd_t(T),njacks),nmr);
    vvvd_t jP5P5_S0(vvd_t(vd_t(T),njacks),nmr);
    
    vvvd_t jP5P5_S(vvd_t(vd_t(T),njacks),nmr);

    //define deltam_cr
    vvvvd_t v_deltamc(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm);
    vvvvd_t v_deltamu(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nr),nm);
    
    // load correlators
#pragma omp parallel for
    for(int mr=0;mr<_nmr;mr++)
        {
            int r = mr%nr;
            int m = (mr-r)/nr;
            
            //load V0P5 correlator
            jV0P5_LO[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            
            //load V0P5 corrections
            jV0P5_LL[mr]=get_contraction("",out_hadr,m,m,r,r,_F ,_F ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0M[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_FF,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0T[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_T ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0P[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_P ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_0S[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_S ,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_M0[mr]=get_contraction("",out_hadr,m,m,r,r,_FF,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_T0[mr]=get_contraction("",out_hadr,m,m,r,r,_T ,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_P0[mr]=get_contraction("",out_hadr,m,m,r,r,_P ,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            jV0P5_S0[mr]=get_contraction("",out_hadr,m,m,r,r,_S ,_LO,"V0P5",IM,UNK,conf_id,path_to_ens);
            
            //load P5P5 correlator
            jP5P5_LO[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            
            //load P5P5 corrections
            jP5P5_LL[mr]=get_contraction("",out_hadr,m,m,r,r,_F ,_F ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0M[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_FF,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0T[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_T ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0P[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_P ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_0S[mr]=get_contraction("",out_hadr,m,m,r,r,_LO,_S ,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_M0[mr]=get_contraction("",out_hadr,m,m,r,r,_FF,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_T0[mr]=get_contraction("",out_hadr,m,m,r,r,_T ,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_P0[mr]=get_contraction("",out_hadr,m,m,r,r,_P ,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
            jP5P5_S0[mr]=get_contraction("",out_hadr,m,m,r,r,_S ,_LO,"P5P5",RE,UNK,conf_id,path_to_ens);
        }
    
#pragma omp parallel for collapse (2)
        for(int mr=0;mr<_nmr;mr++)
            for (int ijack=0; ijack<njacks; ijack++)
            {
                //collect V0P5 corrections
                jV0P5_QED[mr][ijack] =  jV0P5_LL[mr][ijack] +
                                        jV0P5_0M[mr][ijack] +
                                        jV0P5_0T[mr][ijack] +
                                        jV0P5_M0[mr][ijack] +
                                        jV0P5_T0[mr][ijack];
                
                jV0P5_P[mr][ijack]   =  jV0P5_0P[mr][ijack] +
                                        jV0P5_P0[mr][ijack];
                
                jV0P5_S[mr][ijack]   =  jV0P5_0S[mr][ijack] +
                                        jV0P5_S0[mr][ijack];
                
                //collect P5P5 corrections
                jP5P5_QED[mr][ijack] =  jP5P5_LL[mr][ijack] +
                                        jP5P5_0M[mr][ijack] +
                                        jP5P5_0T[mr][ijack] +
                                        jP5P5_M0[mr][ijack] +
                                        jP5P5_T0[mr][ijack];
                
                jP5P5_P[mr][ijack]   =  jP5P5_0P[mr][ijack] +
                                        jP5P5_P0[mr][ijack];
                
                jP5P5_S[mr][ijack]   =  jP5P5_0S[mr][ijack] +
                                        jP5P5_S0[mr][ijack];
            }
    
    
    vvvd_t mean_value_mu(vvd_t(vd_t(0.0,T/2+1),nr),nm), sqr_mean_value_mu(vvd_t(vd_t(0.0,T/2+1),nr),nm), error_mu(vvd_t(vd_t(0.0,T/2+1),nr),nm);
    vvvd_t mean_value_mc(vvd_t(vd_t(0.0,T/2+1),nr),nm), sqr_mean_value_mc(vvd_t(vd_t(0.0,T/2+1),nr),nm), error_mc(vvd_t(vd_t(0.0,T/2+1),nr),nm);
        
    vd_t A(T/2+1),B(T/2+1),C(T/2+1),D(T/2+1),E(T/2+1),F(T/2+1);
    
    // Solving with Kramer:
    //   delta(mPCAC):                  a + b*deltamu + c*deltamcr + (correction to denominator) = 0
    //   delta(slope[P5P5_ins/P5P5]):   d + e*deltamu + f*deltamcr = 0
    
//#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m=0;m<nm;m++)
            for(int r=0;r<nr;r++)
                {
                    int mr = r+nr*m;
                    
                    A = symmetrize(symmetric_derivative(jV0P5_QED[mr][ijack])/jP5P5_LO[mr][ijack] - symmetric_derivative(jV0P5_LO[mr][ijack])*jP5P5_QED[mr][ijack]/jP5P5_LO[mr][ijack]/jP5P5_LO[mr][ijack],1);
                    B = symmetrize(symmetric_derivative(jV0P5_S[mr][ijack])/jP5P5_LO[mr][ijack] - symmetric_derivative(jV0P5_LO[mr][ijack])*jP5P5_S[mr][ijack]/jP5P5_LO[mr][ijack]/jP5P5_LO[mr][ijack],1);
                    C = symmetrize(symmetric_derivative(jV0P5_P[mr][ijack])/jP5P5_LO[mr][ijack] - symmetric_derivative(jV0P5_LO[mr][ijack])*jP5P5_P[mr][ijack]/jP5P5_LO[mr][ijack]/jP5P5_LO[mr][ijack],1);
                    
                    D = effective_slope(symmetrize(jP5P5_QED[mr][ijack]/jP5P5_LO[mr][ijack],1),eff_mass_time[mr][mr][ijack],T/2);
                    E = effective_slope(symmetrize(jP5P5_S[mr][ijack]/jP5P5_LO[mr][ijack],1),eff_mass_time[mr][mr][ijack],T/2);
                    F = effective_slope(symmetrize(jP5P5_P[mr][ijack]/jP5P5_LO[mr][ijack],1),eff_mass_time[mr][mr][ijack],T/2);
                    
                    vd_t den = B*F-C*E;
                    vd_t deltamu = (-A*F+C*D)/den;
                    vd_t deltamc = (-B*D+A*E)/den;
                    
                    v_deltamu[m][r][ijack] = deltamu;
                    v_deltamc[m][r][ijack] = deltamc;
                }
    
#pragma omp parallel for collapse(3)
    for(int t=0;t<T/2/*+1*/;t++)
        for(int r=0;r<nr;r++)
            for(int m=0;m<nm;m++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mean_value_mu[m][r][t] += v_deltamu[m][r][ijack][t]/njacks;
                    mean_value_mc[m][r][t] += v_deltamc[m][r][ijack][t]/njacks;
                    
                    sqr_mean_value_mu[m][r][t] += v_deltamu[m][r][ijack][t]*v_deltamu[m][r][ijack][t]/njacks;
                    sqr_mean_value_mc[m][r][t] += v_deltamc[m][r][ijack][t]*v_deltamc[m][r][ijack][t]/njacks;
                }
                
                error_mu[m][r][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mu[m][r][t]-mean_value_mu[m][r][t]*mean_value_mu[m][r][t]));
                error_mc[m][r][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mc[m][r][t]-mean_value_mc[m][r][t]*mean_value_mc[m][r][t]));
            }
    
    vvd_t coord(vd_t(0.0,T/2+1),1);
    for(int j=0; j<T/2+1; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvd_t jdeltamc(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nr),nm);
    vvvvd_t jdeltamu(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nr),nm);
    
    for(int m=0;m<nm;m++)
        for(int r=0;r<nr;r++)
        {
            jdeltamc[m][r] = polyfit(coord, 1, error_mc[m][r], v_deltamc[m][r], delta_tmin, delta_tmax);
            jdeltamu[m][r] = polyfit(coord, 1, error_mu[m][r], v_deltamu[m][r], delta_tmin, delta_tmax);
        }
    
    vvvd_t deltamc(vvd_t(vd_t(0.0,nr),nm),njacks);
    vvvd_t deltamu(vvd_t(vd_t(0.0,nr),nm),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m=0;m<nm;m++)
            for(int r=0;r<nr;r++)
            {
                deltamc[ijack][m][r]=jdeltamc[m][r][ijack][0];
                deltamu[ijack][m][r]=jdeltamu[m][r][ijack][0];
            }
    
    ofstream outfile_mu, outfile_mc;
    outfile_mc.open(path_to_ens+"deltam_cr_array", ios::out | ios::binary);
    outfile_mu.open(path_to_ens+"deltamu_array", ios::out | ios::binary);
    
    if (outfile_mc.is_open() and outfile_mu.is_open())
    {
        for(int m=0;m<nm;m++)
            for(int r=0;r<nr;r++)
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    outfile_mc.write((char*) &deltamc[ijack][m][r],sizeof(double));
                    outfile_mu.write((char*) &deltamu[ijack][m][r],sizeof(double));
                }
        
        outfile_mu.close();
        outfile_mc.close();
    }
    else cerr<< "Unable to create the output file \"deltam_cr_array\" and \"deltamu_array\" "<<endl;
    
}
