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
#include "print.hpp"
#include "sigmas.hpp"

// compute deltam_cr and deltamu from propagators
void oper_t::compute_deltam_from_prop()
{
    using namespace sigma;
    
    ifstream deltamu_data;
    ifstream deltamc_data;
    deltamu_data.open(path_to_ens+"deltamu_array");
    deltamc_data.open(path_to_ens+"deltam_cr_array");
    
    if(deltamu_data.good() and deltamc_data.good() and !deltam_computed)
    {
        cout<<"Reading deltam from files: \""<<path_to_ens<<"deltam*\""<<endl<<endl;
        read_vec_bin(deltamu,path_to_ens+"deltamu_array");
        read_vec_bin(deltam_cr,path_to_ens+"deltam_cr_array");
    }
    else
    {
        if(!deltam_computed)
            cout<<"Computing deltam"<<endl;
        else
            cout<<"Recomputing deltam"<<endl;

        //define deltam_cr
        vvvd_t v_deltamc(vvd_t(vd_t(0.0,_linmoms),njacks),_nmr);
        vvvd_t v_deltamu(vvd_t(vd_t(0.0,_linmoms),njacks),_nmr);
        
        vvd_t   mean_value_mu(vd_t(0.0,_linmoms),_nmr),
        sqr_mean_value_mu=mean_value_mu,
        error_mu=mean_value_mu;
        vvd_t   mean_value_mc(vd_t(0.0,_linmoms),_nmr),
        sqr_mean_value_mc=mean_value_mc,
        error_mc=mean_value_mc;
        
        // Solving with Kramer:
        //   B*(deltamu) + C*(deltamcr) - A = 0
        //   E*(deltamu) + F*(deltamcr) - D = 0
        
#pragma omp parallel for collapse(3)
        for(int imom=0;imom<_linmoms;imom++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr=0;mr<_nmr;mr++)
                {
                    double A = sigma[imom][SIGMA2][PH][ijack][mr];
                    double B = sigma[imom][SIGMA2][S ][ijack][mr];
                    double C = sigma[imom][SIGMA2][P ][ijack][mr];
                    
                    double D = sigma[imom][SIGMA3][PH][ijack][mr];
                    double E = sigma[imom][SIGMA3][S ][ijack][mr];
                    double F = sigma[imom][SIGMA3][P ][ijack][mr];
                    
                    double den = B*F-C*E;
                    double delta_mu = (-A*F+C*D)/den;
                    double delta_mc = (-B*D+A*E)/den;
                    
                    v_deltamu[mr][ijack][imom] = delta_mu;
                    v_deltamc[mr][ijack][imom] = delta_mc;
                }
        
#pragma omp parallel for collapse(2)
        for(int imom=0;imom<_linmoms;imom++)
            for(int mr=0;mr<_nmr;mr++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mean_value_mu[mr][imom] += v_deltamu[mr][ijack][imom]/njacks;
                    mean_value_mc[mr][imom] += v_deltamc[mr][ijack][imom]/njacks;
                    
                    sqr_mean_value_mu[mr][imom] += v_deltamu[mr][ijack][imom]*v_deltamu[mr][ijack][imom]/njacks;
                    sqr_mean_value_mc[mr][imom] += v_deltamc[mr][ijack][imom]*v_deltamc[mr][ijack][imom]/njacks;
                }
                
                error_mu[mr][imom] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mu[mr][imom]-mean_value_mu[mr][imom]*mean_value_mu[mr][imom]));
                error_mc[mr][imom] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mc[mr][imom]-mean_value_mc[mr][imom]*mean_value_mc[mr][imom]));
            }
        
        int npar=4;
        vvd_t coord(vd_t(0.0,_linmoms),npar);
        for(int j=0; j<_linmoms; j++)
        {
            coord[0][j] = 1.0;
            coord[1][j] = p2[j];
            coord[2][j] = p2[j]*p2[j];
            coord[3][j] = p2[j]*p2[j]*p2[j];
        }
        
        double p2_min = 0.01;
        double p2_max = 2.0;
        
        for(int mr=0;mr<_nmr;mr++)
        {
            vvd_t deltamc_tmp = polyfit(coord,npar,error_mc[mr],v_deltamc[mr],p2_min,p2_max);
            vvd_t deltamu_tmp = polyfit(coord,npar,error_mu[mr],v_deltamu[mr],p2_min,p2_max);
            
            for(int ijack=0;ijack<njacks;ijack++)
            {
                deltam_cr[ijack][mr]=deltamc_tmp[ijack][0];
                deltamu[ijack][mr]=deltamu_tmp[ijack][0];
            }
        }
    }
    
    if(!deltam_computed)
    {
        print_vec_bin(deltam_cr,path_to_ens+"deltam_cr_array");
        print_vec_bin(deltamu,path_to_ens+"deltamu_array");
    }
    
    vd_t deltamu_ave(0.0,_nmr), deltamc_ave(0.0,_nmr);
    vd_t sqr_deltamu_ave(0.0,_nmr), sqr_deltamc_ave(0.0,_nmr);
    vd_t deltamu_err(0.0,_nmr), deltamc_err(0.0,_nmr);
    
    for(int mr=0;mr<_nmr;mr++)
    {
        for(int ijack=0;ijack<njacks;ijack++)
        {
            deltamu_ave[mr] += deltamu[ijack][mr]/njacks;
            deltamc_ave[mr] += deltam_cr[ijack][mr]/njacks;
            
            sqr_deltamu_ave[mr] += deltamu[ijack][mr]*deltamu[ijack][mr]/njacks;
            sqr_deltamc_ave[mr] += deltam_cr[ijack][mr]*deltam_cr[ijack][mr]/njacks;
        }
        
        deltamu_err[mr] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_deltamu_ave[mr]-deltamu_ave[mr]*deltamu_ave[mr]));
        deltamc_err[mr] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_deltamc_ave[mr]-deltamc_ave[mr]*deltamc_ave[mr]));
    }
    
    for(int mr=0;mr<_nmr;mr++)
        printf("mr: %d \t %lg +- %lg \t %lg +- %lg \n",mr,deltamc_ave[mr],deltamc_err[mr],deltamu_ave[mr],deltamu_err[mr]);

}


// compute deltam_cr and deltamu from correlators
void oper_t::compute_deltam()
{
    ifstream deltamu_data;
    ifstream deltamc_data;
    deltamu_data.open(path_to_ens+"deltamu_array");
    deltamc_data.open(path_to_ens+"deltam_cr_array");
    
    if(deltamu_data.good() and deltamc_data.good() and !deltam_computed)
    {
        cout<<"Reading deltam from files"<<endl<<endl;
        read_vec_bin(deltamu,path_to_ens+"deltamu_array");
        read_vec_bin(deltam_cr,path_to_ens+"deltam_cr_array");
    }
    else
    {
        if(!deltam_computed)
            cout<<"Computing deltam"<<endl;
        else
            cout<<"Recomputing deltam"<<endl;
        
        // array of the configurations
        int conf_id[nconfs];
        for(int iconf=0;iconf<nconfs;iconf++)
            conf_id[iconf]=conf_init+iconf*conf_step;
        
        int T=size[0];
        
        //define jackknife V0P5 correlators
        vvvd_t jV0P5_LO(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jV0P5_LL(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jV0P5_0M(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jV0P5_M0(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jV0P5_0T(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jV0P5_T0(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jV0P5_QED(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jV0P5_0P(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jV0P5_P0(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jV0P5_P(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jV0P5_0S(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jV0P5_S0(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jV0P5_S(vvd_t(vd_t(T),njacks),_nmr);
        
        //define jackknife P5P5 correlators
        vvvd_t jP5P5_LO(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jP5P5_LL(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jP5P5_0M(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jP5P5_M0(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jP5P5_0T(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jP5P5_T0(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jP5P5_QED(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jP5P5_0P(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jP5P5_P0(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jP5P5_P(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jP5P5_0S(vvd_t(vd_t(T),njacks),_nmr);
        vvvd_t jP5P5_S0(vvd_t(vd_t(T),njacks),_nmr);
        
        vvvd_t jP5P5_S(vvd_t(vd_t(T),njacks),_nmr);
        
        //define deltam_cr
        vvvd_t v_deltamc(vvd_t(vd_t(0.0,T/2+1),njacks),_nmr);
        vvvd_t v_deltamu(vvd_t(vd_t(0.0,T/2+1),njacks),_nmr);
        
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
        
        
        vvd_t mean_value_mu(vd_t(0.0,T/2+1),_nmr), sqr_mean_value_mu(vd_t(0.0,T/2+1),_nmr), error_mu(vd_t(0.0,T/2+1),_nmr);
        vvd_t mean_value_mc(vd_t(0.0,T/2+1),_nmr), sqr_mean_value_mc(vd_t(0.0,T/2+1),_nmr), error_mc(vd_t(0.0,T/2+1),_nmr);
        
        vd_t A(T/2+1),B(T/2+1),C(T/2+1),D(T/2+1),E(T/2+1),F(T/2+1);
        
        // Solving with Kramer:
        //   delta(mPCAC):                  a + b*deltamu + c*deltamcr + (correction to denominator) = 0
        //   delta(slope[P5P5_ins/P5P5]):   d + e*deltamu + f*deltamcr = 0
        
        //#pragma omp parallel for collapse(3)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m=0;m<_nm;m++)
                for(int r=0;r<_nr;r++)
                {
                    int mr = r+nr*m;
                    
                    A = symmetrize(symmetric_derivative(jV0P5_QED[mr][ijack])/jP5P5_LO[mr][ijack] - symmetric_derivative(jV0P5_LO[mr][ijack])*jP5P5_QED[mr][ijack]/jP5P5_LO[mr][ijack]/jP5P5_LO[mr][ijack],1);
                    B = symmetrize(symmetric_derivative(jV0P5_S[mr][ijack])/jP5P5_LO[mr][ijack] - symmetric_derivative(jV0P5_LO[mr][ijack])*jP5P5_S[mr][ijack]/jP5P5_LO[mr][ijack]/jP5P5_LO[mr][ijack],1);
                    C = symmetrize(symmetric_derivative(jV0P5_P[mr][ijack])/jP5P5_LO[mr][ijack] - symmetric_derivative(jV0P5_LO[mr][ijack])*jP5P5_P[mr][ijack]/jP5P5_LO[mr][ijack]/jP5P5_LO[mr][ijack],1);
                    
                    D = effective_slope(symmetrize(jP5P5_QED[mr][ijack]/jP5P5_LO[mr][ijack],1),eff_mass_time[m][m][ijack],T/2);
                    E = effective_slope(symmetrize(jP5P5_S[mr][ijack]/jP5P5_LO[mr][ijack],1),eff_mass_time[m][m][ijack],T/2);
                    F = effective_slope(symmetrize(jP5P5_P[mr][ijack]/jP5P5_LO[mr][ijack],1),eff_mass_time[m][m][ijack],T/2);
                    
                    vd_t den = B*F-C*E;
                    vd_t deltamu = (-A*F+C*D)/den;
                    vd_t deltamc = (-B*D+A*E)/den;
                    
                    v_deltamu[mr][ijack] = deltamu;
                    v_deltamc[mr][ijack] = deltamc;
                }
        
#pragma omp parallel for collapse(2)
        for(int t=0;t<T/2/*+1*/;t++)
            for(int mr=0;mr<_nmr;mr++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mean_value_mu[mr][t] += v_deltamu[mr][ijack][t]/njacks;
                    mean_value_mc[mr][t] += v_deltamc[mr][ijack][t]/njacks;
                    
                    sqr_mean_value_mu[mr][t] += v_deltamu[mr][ijack][t]*v_deltamu[mr][ijack][t]/njacks;
                    sqr_mean_value_mc[mr][t] += v_deltamc[mr][ijack][t]*v_deltamc[mr][ijack][t]/njacks;
                }
                
                error_mu[mr][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mu[mr][t]-mean_value_mu[mr][t]*mean_value_mu[mr][t]));
                error_mc[mr][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mean_value_mc[mr][t]-mean_value_mc[mr][t]*mean_value_mc[mr][t]));
            }
        
        vvd_t coord(vd_t(0.0,T/2+1),1);
        for(int j=0; j<T/2+1; j++)
        {
            coord[0][j] = 1.0;  //fit a costante
        }
        
        vvvd_t jdeltamc(vvd_t(vd_t(0.0,coord.size()),njacks),_nmr);
        vvvd_t jdeltamu(vvd_t(vd_t(0.0,coord.size()),njacks),_nmr);
        
        for(int mr=0;mr<_nmr;mr++)
        {
            jdeltamc[mr] = polyfit(coord, 1, error_mc[mr], v_deltamc[mr], delta_tmin, delta_tmax);
            jdeltamu[mr] = polyfit(coord, 1, error_mu[mr], v_deltamu[mr], delta_tmin, delta_tmax);
        }
        
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<_nmr;mr++)
            {
                deltam_cr[ijack][mr]=jdeltamc[mr][ijack][0];
                deltamu[ijack][mr]=jdeltamu[mr][ijack][0];
            }
    }
    
    if(!deltam_computed)
    {
        print_vec_bin(deltam_cr,path_to_ens+"deltam_cr_array");
        print_vec_bin(deltamu,path_to_ens+"deltamu_array");
        
        vd_t deltamu_ave(0.0,_nmr), deltamc_ave(0.0,_nmr);
        vd_t sqr_deltamu_ave(0.0,_nmr), sqr_deltamc_ave(0.0,_nmr);
        vd_t deltamu_err(0.0,_nmr), deltamc_err(0.0,_nmr);
        
        for(int mr=0;mr<_nmr;mr++)
        {
            for(int ijack=0;ijack<njacks;ijack++)
            {
                deltamu_ave[mr] += deltamu[ijack][mr]/njacks;
                deltamc_ave[mr] += deltam_cr[ijack][mr]/njacks;
                
                sqr_deltamu_ave[mr] += deltamu[ijack][mr]*deltamu[ijack][mr]/njacks;
                sqr_deltamc_ave[mr] += deltam_cr[ijack][mr]*deltam_cr[ijack][mr]/njacks;
            }
            
            deltamu_err[mr] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_deltamu_ave[mr]-deltamu_ave[mr]*deltamu_ave[mr]));
            deltamc_err[mr] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_deltamc_ave[mr]-deltamc_ave[mr]*deltamc_ave[mr]));
        }
        
        for(int mr=0;mr<_nmr;mr++)
            printf("mr: %d \t %lg +- %lg \t %lg +- %lg \n",mr,deltamc_ave[mr],deltamc_err[mr],deltamu_ave[mr],deltamu_err[mr]);
    }
}
