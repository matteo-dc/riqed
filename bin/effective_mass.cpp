#include "global.hpp"
#include "aliases.hpp"
#include "contractions.hpp"
#include "fit.hpp"
#include <iostream>
#include <omp.h>
#include "operations.hpp"
#include "tools.hpp"
#include "ave_err.hpp"

//function to use in Newton's method for M_eff
double f_mass (int t, int T, double x0, double y)
{
    double f = cosh(x0*(t-T/2))/cosh(x0*(t+1-T/2)) - y;  // y=c(t)/c(t+1), where c(t) is the correlator at the time t
    
    return f;
}

//derivative to use in Newton's method for M_eff
double f_prime_mass (int t, int T, double x0)
{
    int k = t-T/2;
    
    double fp = ( k*sinh(x0*k) - (1+k)*cosh(x0*k)*tanh(x0*(1+k)) )/cosh(x0*(t+1-T/2));
    
    return fp;
}

//Newton's Method for M_eff (in a fixed jackknife)
double solve_Newton (vvd_t C, int ijack, int t, const int T)
{
    double k = C[ijack][t]/C[ijack][t+1];
    
    // cout<<"**********DEBUG*************"<<endl;
    // cout<<"jack: "<<ijack<<"  t: "<<t<<"  c(t)/c(t+1): "<<k<<endl;
    // cout<<"**********DEBUG*************"<<endl;
    
    if(k<1.0)
    {return nan("");}
    else{
        
        double eps=1e-14;
        int max_iteration=500;
        int count_iteration=0;
        int g=0;
        
        double x0=1.09; //seed
        double x1;
        
        double y, yp, x;
        
        x1=x0;
        do
        {
            x=x1;
            
            y=f_mass(t,T,x,k);
            yp=f_prime_mass(t,T,x);
            
            x1 = x - y/yp;
            
            count_iteration++;
            g++;
            
            //  cout<<count_iteration<<endl;
            
        } while ( abs(x1-x) >= x1*eps and count_iteration!=max_iteration );
        
        
        // cout<<x0<<" ";
        
        // cout<<"********DEBUG*****************************"<<endl;
        // if(count_iteration==max_iteration)
        //   cerr<<t<<" Newton's method did not converge for the jackknife n. "<<ijack<<" in "<<max_iteration<<" iterations. The value is "<<x1<<" k "<<k<<endl;
        // else cout<<t<<" Jackknife n. "<<ijack<<" has converged with success to the value "<<x1<<" in "<<g<<" iterations"<<" k "<<k<<endl;
        // cout<<"********DEBUG*****************************"<<endl;
        
        return x1;
    }
}

// compute mPCAC
void oper_t::compute_mPCAC(const string &suffix)
{
    cout<<"Computing mPCAC "<<suffix<<endl<<endl;
    
    int _nm=0, _nr=0;
    if(suffix.compare("")==0)
    {
        _nm=nm;
    }
    else if(suffix.compare("sea")==0)
    {
        _nm=1;
    }
    
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    // define jackknife correlators
    vvvvvd_t jP5P5_00(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),_nr),_nm),_nm);
    vvvvvd_t jV0P5_00(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),_nr),_nm),_nm);
    
    vvvvd_t jV0P5_00_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),_nm),_nm);
    vvvvd_t jP5P5_00_ave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),_nm),_nm);
    
    vvvvd_t mPCAC_corr(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),_nm),_nm);
    vvvvd_t mPCAC_corr_symm(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),_nm),_nm);
    
    vvvd_t mPCAC_corr_ave(vvd_t(vd_t(0.0,T/2+1),_nm),_nm);
    vvvd_t mPCAC_corr_symm_ave(vvd_t(vd_t(0.0,T/2+1),_nm),_nm);
    vvvd_t sqr_mPCAC_corr_ave(vvd_t(vd_t(0.0,T/2+1),_nm),_nm);
    vvvd_t sqr_mPCAC_corr_symm_ave(vvd_t(vd_t(0.0,T/2+1),_nm),_nm);
    vvvd_t mPCAC_corr_err(vvd_t(vd_t(0.0,T/2+1),_nm),_nm);
    vvvd_t mPCAC_corr_symm_err(vvd_t(vd_t(0.0,T/2+1),_nm),_nm);

    // load correlators
#pragma omp parallel for collapse (3)
    for(int m_fw=0;m_fw<_nm;m_fw++)
        for(int m_bw=0;m_bw<_nm;m_bw++)
            for(int r=0;r<nr;r++)
                {
                    jP5P5_00[m_fw][m_bw][r]=
                        get_contraction(suffix,out_hadr,m_fw,m_bw,r,r,_LO,_LO,"P5P5",RE,EVN,conf_id ,path_to_ens);
                    jV0P5_00[m_fw][m_bw][r]=
                        get_contraction(suffix,out_hadr,m_fw,m_bw,r,r,_LO,_LO,"V0P5",IM,ODD,conf_id,path_to_ens);
                }
    
    //average over r (applying r-parity)
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<_nm;m_fw++)
        for(int m_bw=0;m_bw<_nm;m_bw++)
            for (int ijack=0; ijack<njacks; ijack++)
                for(int t=0;t<T/2+1;t++)
                    for(int r1=0;r1<nr;r1++)
                    {
                        int r2=(r1+1)%nr;
                        
                        int cr_evn = r2 + r1;
                        int cr_odd = r2 - r1;
                        
                        // taking the same r
                        jP5P5_00_ave[m_fw][m_bw][ijack][t] += jP5P5_00[m_fw][m_bw][r1][ijack][t]*cr_evn/nr;
                        jV0P5_00_ave[m_fw][m_bw][ijack][t] += jV0P5_00[m_fw][m_bw][r1][ijack][t]*cr_odd/nr;
                    }

    //define mPCAC with forward and symmetric derivative
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<_nm;m_fw++)
        for(int m_bw=0;m_bw<_nm;m_bw++)
            for (int ijack=0; ijack<njacks; ijack++)
                for(int t=0;t<T/2+1;t++)
                    {
                        mPCAC_corr[m_fw][m_bw][ijack][t] = forward_derivative(jV0P5_00_ave[m_fw][m_bw][ijack])[t]/(2.0*jP5P5_00_ave[m_fw][m_bw][ijack][t]);
                        mPCAC_corr_symm[m_fw][m_bw][ijack][t] = symmetric_derivative(jV0P5_00_ave[m_fw][m_bw][ijack])[t]/(2.0*jP5P5_00_ave[m_fw][m_bw][ijack][t]);
                    }
    
#pragma omp parallel for collapse(3)
    for(int m_fw=0;m_fw<_nm;m_fw++)
        for(int m_bw=0;m_bw<_nm;m_bw++)
            for(int t=0;t<T/2;t++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mPCAC_corr_ave[m_fw][m_bw][t]+=mPCAC_corr[m_fw][m_bw][ijack][t]/njacks;
                    sqr_mPCAC_corr_ave[m_fw][m_bw][t]+=mPCAC_corr[m_fw][m_bw][ijack][t]*mPCAC_corr[m_fw][m_bw][ijack][t]/njacks;
                    
                    mPCAC_corr_symm_ave[m_fw][m_bw][t]+=mPCAC_corr_symm[m_fw][m_bw][ijack][t]/njacks;
                    sqr_mPCAC_corr_symm_ave[m_fw][m_bw][t]+=mPCAC_corr_symm[m_fw][m_bw][ijack][t]*mPCAC_corr_symm[m_fw][m_bw][ijack][t]/njacks;
                }
                
                mPCAC_corr_err[m_fw][m_bw][t]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_mPCAC_corr_ave[m_fw][m_bw][t]-mPCAC_corr_ave[m_fw][m_bw][t]*mPCAC_corr_ave[m_fw][m_bw][t]));
                mPCAC_corr_symm_err[m_fw][m_bw][t]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_mPCAC_corr_symm_ave[m_fw][m_bw][t]-mPCAC_corr_symm_ave[m_fw][m_bw][t]*mPCAC_corr_symm_ave[m_fw][m_bw][t]));
            }

    
    //t-range for the fit
    int t_min = delta_tmin;
    int t_max = delta_tmax;
    
    vvd_t coord(vd_t(0.0,T/2),1);
    for(int j=0; j<T/2; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvd_t mPCAC_tmp(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),_nm),_nm);
    vvvvd_t mPCAC_symm_tmp(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),_nm),_nm);
    
    for(int m_fw=0;m_fw<_nm;m_fw++)
        for(int m_bw=0;m_bw<_nm;m_bw++)
        {
            mPCAC_tmp[m_fw][m_bw] = polyfit(coord, 1, mPCAC_corr_err[m_fw][m_bw], mPCAC_corr[m_fw][m_bw], t_min, t_max);
            mPCAC_symm_tmp[m_fw][m_bw] = polyfit(coord, 1, mPCAC_corr_symm_err[m_fw][m_bw], mPCAC_corr_symm[m_fw][m_bw], t_min, t_max);
        }
    
    vvvd_t mPCAC(vvd_t(vd_t(0.0,_nm),_nm),njacks);
    vvvd_t mPCAC_symm(vvd_t(vd_t(0.0,_nm),_nm),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<_nm;m_fw++)
            for(int m_bw=0;m_bw<_nm;m_bw++)
            {
                mPCAC[ijack][m_fw][m_bw]=mPCAC_tmp[m_fw][m_bw][ijack][0];
                mPCAC_symm[ijack][m_fw][m_bw]=mPCAC_symm_tmp[m_fw][m_bw][ijack][0];
            }
    
    for(int m=0;m<_nm;m++)
    {
        string mass;
        if(_nm==1) mass="sea";
        else mass=to_string(m);
        cout<<"mPCAC["<<mass<<"] = "<<get<0>(ave_err(mPCAC))[m][m]<<" "<<get<1>(ave_err(mPCAC))[m][m]<<endl;
        cout<<"mPCAC_symm["<<mass<<"] = "<<get<0>(ave_err(mPCAC_symm))[m][m]<<" "<<get<1>(ave_err(mPCAC_symm))[m][m]<<endl;
    }

    string suffix_new=suffix;
    if(suffix.compare("sea")==0) suffix_new="_"+suffix;
    
    cout<<endl;
    ofstream outfile;
    outfile.open(path_to_ens+"mPCAC"+suffix_new, ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m_fw=0;m_fw<_nm;m_fw++)
                for(int m_bw=0;m_bw<_nm;m_bw++)
                {
                    outfile.write((char*) &mPCAC_symm[ijack][m_fw][m_bw],sizeof(double));
                }
        
        outfile.close();
    }
    else cerr<<"Unable to create the output file \"mPCAC"<<suffix<<"\" "<<endl;
}

// compute effective meson mass
void oper_t::compute_eff_mass()
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    // define jackknife P5P5 correlators
    vvvvvd_t jP5P5_00(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nr),nm),nm);
    
    // load correlators
#pragma omp parallel for collapse (3)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int r=0;r<nr;r++)
                jP5P5_00[m_fw][m_bw][r]=get_contraction("",out_hadr,m_fw,m_bw,r,r,_LO,_LO,"P5P5",RE,EVN,conf_id,path_to_ens);

    // define jackknife P5P5 correlators r-averaged
    vvvvd_t jP5P5_00_rave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int ijack=0; ijack<njacks;ijack++)
                for(int t=0;t<T/2+1;t++)
                    for(int r=0;r<nr;r++)
                        jP5P5_00_rave[m_fw][m_bw][ijack][t]+=jP5P5_00[m_fw][m_bw][r][ijack][t]/nr;
    
    // compute effective mass time array
    vvvvd_t M_eff(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nm),nm);
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int ijack=0; ijack<njacks;ijack++)
                for(int t=0;t<T/2;t++)
                    M_eff[m_fw][m_bw][ijack][t] =
                        solve_Newton (jP5P5_00_rave[m_fw][m_bw],ijack,t,T);
    
    vvvd_t mass_ave(vvd_t(vd_t(0.0,T/2),nm),nm);
    vvvd_t sqr_mass_ave=mass_ave;
    vvvd_t mass_err=mass_ave;
    
#pragma omp parallel for collapse(3)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int t=0;t<T/2;t++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mass_ave[m_fw][m_bw][t] += M_eff[m_fw][m_bw][ijack][t]/njacks;
                    sqr_mass_ave[m_fw][m_bw][t] +=
                        M_eff[m_fw][m_bw][ijack][t]*M_eff[m_fw][m_bw][ijack][t]/njacks;
                }
                
                mass_err[m_fw][m_bw][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_mass_ave[m_fw][m_bw][t]-mass_ave[m_fw][m_bw][t]*mass_ave[m_fw][m_bw][t]));
            }
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
        {
            ofstream outfile_m;
            outfile_m.open(path_to_ens+"plots/eff_mass_"+to_string(m_fw)+"_"+to_string(m_bw)+".txt");
            
            if (outfile_m.is_open())
            {
                for(int t=0;t<T/2;t++)
                        {
                            outfile_m<<t<<"\t"<<mass_ave[m_fw][m_bw][t]<<"\t"<<mass_err[m_fw][m_bw][t]<<endl;
                        }
                
                outfile_m.close();
            }
            else cerr<<"Unable to create the output file \"plots/eff_mass_"<<m_fw<<"_"<<m_bw<<".txt"<<"\" "<<endl;
        }
    
    //t-range for the fit
    int t_min = delta_tmin;
    int t_max = delta_tmax;
    
    vvd_t coord(vd_t(0.0,T/2),1);
    for(int j=0; j<T/2; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvd_t jeff_mass(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nm),nm);
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            jeff_mass[m_fw][m_bw] =
                polyfit(coord,1,mass_err[m_fw][m_bw],M_eff[m_fw][m_bw],t_min,t_max);
    
    // define meson effective mass
    vvvd_t eff_mass_tmp(vvd_t(vd_t(0.0,nm),nm),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                eff_mass_tmp[ijack][m_fw][m_bw] = jeff_mass[m_fw][m_bw][ijack][0];
    
    
    ofstream outfile;
    outfile.open(path_to_ens+"eff_mass_array", ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m_fw=0;m_fw<nm;m_fw++)
                for(int m_bw=0;m_bw<nm;m_bw++)
                {
                    outfile.write((char*) &eff_mass_tmp[ijack][m_fw][m_bw],sizeof(double));
                }
        
        outfile.close();
    }
    else cerr<<"Unable to create the output file \"eff_mass_array\" "<<endl;
    
    ofstream outfile_time;
    outfile_time.open(path_to_ens+"eff_mass_array_time", ios::out | ios::binary);
    
    if (outfile_time.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m_fw=0;m_fw<nm;m_fw++)
                for(int m_bw=0;m_bw<nm;m_bw++)
                    for(int t=0;t<T/2;t++)
                    {
                        outfile_time.write((char*) &M_eff[m_fw][m_bw][ijack][t],sizeof(double));
                    }
        
        outfile_time.close();
    }
    else cerr<<"Unable to create the output file \"eff_mass_array_time\" "<<endl;
}

// compute meson mass correction
void oper_t::compute_eff_mass_correction()
{
    if(ntypes!=3)
    {
        cout<<"Not implemented for ntypes!=3"<<endl;
        exit(0);
    }
    else
    {
        // array of the configurations
        int conf_id[nconfs];
        for(int iconf=0;iconf<nconfs;iconf++)
            conf_id[iconf]=conf_init+iconf*conf_step;
        
        int T=size[0];
        
        // define jackknife P5P5 correlators
        vvvvvd_t jP5P5_00(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nr),nm),nm);
        vvvvvd_t jP5P5_LL(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nr),nm),nm);
        vvvvvd_t jP5P5_0M(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nr),nm),nm);
        vvvvvd_t jP5P5_M0(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nr),nm),nm);
        // define jackknife complete correction
        vvvvvd_t jP5P5_QED(vvvvd_t(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nr),nm),nm);
        
        
        // load correlators
#pragma omp parallel for collapse (3)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int r=0;r<nr;r++)
                {
                    jP5P5_00[m_fw][m_bw][r]=get_contraction("",out_hadr,m_fw,m_bw,r,r,_LO,_LO,"P5P5",RE,EVN,conf_id,path_to_ens);
                    jP5P5_LL[m_fw][m_bw][r]=get_contraction("",out_hadr,m_fw,m_bw,r,r,_F,_F,"P5P5",RE,EVN,conf_id,path_to_ens);
                    jP5P5_0M[m_fw][m_bw][r]=get_contraction("",out_hadr,m_fw,m_bw,r,r,_QED,_LO,"P5P5",RE,EVN,conf_id,path_to_ens);
                    jP5P5_M0[m_fw][m_bw][r]=get_contraction("",out_hadr,m_bw,m_fw,r,r,_QED,_LO,"P5P5",RE,EVN,conf_id,path_to_ens);
                    
                    for(int ijack=0; ijack<njacks;ijack++)
                        jP5P5_QED[m_fw][m_bw][r][ijack]=jP5P5_LL[m_fw][m_bw][r][ijack]+
                                                        jP5P5_0M[m_fw][m_bw][r][ijack]+
                                                        jP5P5_M0[m_fw][m_bw][r][ijack];
                }
        
        // define jackknife P5P5 correlators r-averaged
        vvvvd_t jP5P5_00_rave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
        vvvvd_t jP5P5_QED_rave(vvvd_t(vvd_t(vd_t(0.0,T/2+1),njacks),nm),nm);
                
#pragma omp parallel for collapse(4)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int ijack=0; ijack<njacks;ijack++)
                    for(int t=0;t<T/2+1;t++)
                        for(int r=0;r<nr;r++)
                        {
                            jP5P5_00_rave[m_fw][m_bw][ijack][t] +=jP5P5_00[m_fw][m_bw][r][ijack][t]/nr;
                            jP5P5_QED_rave[m_fw][m_bw][ijack][t]+=jP5P5_QED[m_fw][m_bw][r][ijack][t]/nr;
                        }
        
        // compute effective mass time array
        vvvvd_t M_eff(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nm),nm);
        vvvvd_t dM_eff(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nm),nm);
        
        //LO
#pragma omp parallel for collapse(4)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int ijack=0; ijack<njacks;ijack++)
                    for(int t=0;t<T/2;t++)
                        M_eff[m_fw][m_bw][ijack][t] = solve_Newton(jP5P5_00_rave[m_fw][m_bw],ijack,t,T);
        // QED
#pragma omp parallel for collapse(3)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int ijack=0; ijack<njacks;ijack++)
                    dM_eff[m_fw][m_bw][ijack] = - effective_slope(jP5P5_QED_rave[m_fw][m_bw][ijack]/jP5P5_00_rave[m_fw][m_bw][ijack],M_eff[m_fw][m_bw][ijack],T/2);
        
        vvvd_t dM_ave(vvd_t(vd_t(0.0,T/2),nm),nm);
        vvvd_t sqr_dM_ave = dM_ave;
        vvvd_t dM_err = dM_ave;
        
#pragma omp parallel for collapse(3)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                for(int t=0;t<T/2;t++)
                {
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
                        dM_ave[m_fw][m_bw][t] += dM_eff[m_fw][m_bw][ijack][t]/njacks;
                        sqr_dM_ave[m_fw][m_bw][t] += dM_eff[m_fw][m_bw][ijack][t]*dM_eff[m_fw][m_bw][ijack][t]/njacks;
                    }
                    
                    dM_err[m_fw][m_bw][t] = sqrt((double)(njacks-1))*sqrt(fabs(sqr_dM_ave[m_fw][m_bw][t]-dM_ave[m_fw][m_bw][t]*dM_ave[m_fw][m_bw][t]));
                }
        
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
            {
                ofstream outfile_m;
                outfile_m.open(path_to_ens+"plots/eff_mass_corr_"+to_string(m_fw)+"_"+to_string(m_bw)+".txt");
                
                if (outfile_m.is_open())
                {
                    for(int t=0;t<T/2;t++)
                    {
                        outfile_m<<t<<"\t"<<dM_ave[m_fw][m_bw][t]<<"\t"<<dM_err[m_fw][m_bw][t]<<endl;
                    }
                    
                    outfile_m.close();
                }
                else cerr<<"Unable to create the output file \"plots/eff_mass_corr_"<<m_fw<<"_"<<m_bw<<".txt"<<"\" "<<endl;
            }
        
        
        //t-range for the fit
        int t_min = delta_tmin;
        int t_max = delta_tmax;
        
        vvd_t coord(vd_t(0.0,T/2),1);
        for(int j=0; j<T/2; j++)
        {
            coord[0][j] = 1.0;  //fit a costante
        }
        
        vvvvd_t jdM_eff(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nm),nm);
        
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                jdM_eff[m_fw][m_bw] = polyfit(coord,1,dM_err[m_fw][m_bw],dM_eff[m_fw][m_bw],t_min,t_max);
        
        // define meson effective mass
        vvvd_t dM_eff_tmp(vvd_t(vd_t(0.0,nm),nm),njacks);
        
#pragma omp parallel for collapse(3)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m_fw=0;m_fw<nm;m_fw++)
                for(int m_bw=0;m_bw<nm;m_bw++)
                    dM_eff_tmp[ijack][m_fw][m_bw] = jdM_eff[m_fw][m_bw][ijack][0];
        
        
        ofstream outfile;
        outfile.open(path_to_ens+"eff_mass_corr_array", ios::out | ios::binary);
        
        if (outfile.is_open())
        {
            for(int ijack=0;ijack<njacks;ijack++)
                for(int m_fw=0;m_fw<nm;m_fw++)
                    for(int m_bw=0;m_bw<nm;m_bw++)
                    {
                        outfile.write((char*) &dM_eff_tmp[ijack][m_fw][m_bw],sizeof(double));
                    }
            
            outfile.close();
        }
        else cerr<<"Unable to create the output file \"eff_mass_corr_array\" "<<endl;
        
        ofstream outfile_time;
        outfile_time.open(path_to_ens+"eff_mass_corr_array_time", ios::out | ios::binary);
        
        if (outfile_time.is_open())
        {
            for(int ijack=0;ijack<njacks;ijack++)
                for(int m_fw=0;m_fw<nm;m_fw++)
                    for(int m_bw=0;m_bw<nm;m_bw++)
                        for(int t=0;t<T/2;t++)
                        {
                            outfile_time.write((char*) &dM_eff[m_fw][m_bw][ijack][t],sizeof(double));
                        }
            
            outfile_time.close();
        }
        else cerr<<"Unable to create the output file \"eff_mass_corr_array_time\" "<<endl;
    }
}

// compute effective sea mass
void oper_t::compute_eff_mass_sea()
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    // define jackknife P5P5 correlators
    vvvd_t jP5P5_00(vvd_t(vd_t(T/2+1),njacks),nr);
    
    // load correlators
    for(int r=0;r<nr;r++)
        jP5P5_00[r]=get_contraction("sea",out_hadr,0,0,r,r,_LO,_LO,"P5P5",RE,EVN,conf_id,path_to_ens);
    
    // define jackknife P5P5 correlators
    vvd_t jP5P5_00_rave(vd_t(0.0,T/2+1),njacks);
#pragma omp parallel for collapse(2)
    for(int ijack=0; ijack<njacks;ijack++)
        for(int t=0;t<T/2+1;t++)
            for(int r=0;r<nr;r++)
                jP5P5_00_rave[ijack][t]+=jP5P5_00[r][ijack][t]/nr;
    
    // define effective mass array
    vvd_t M_eff(vd_t(T/2+1),njacks);
    
#pragma omp parallel for collapse(2)
    for(int ijack=0; ijack<njacks;ijack++)
        for(int t=0;t<T/2;t++)
            M_eff[ijack][t] = solve_Newton(jP5P5_00_rave,ijack,t,T);
    
    vd_t mass_ave(0.0,T/2);
    vd_t sqr_mass_ave=mass_ave;
    vd_t mass_err=mass_ave;
    
#pragma omp parallel for
    for(int t=0;t<T/2;t++)
    {
        for(int ijack=0;ijack<njacks;ijack++)
        {
            mass_ave[t]+=M_eff[ijack][t]/njacks;
            sqr_mass_ave[t]+=M_eff[ijack][t]*M_eff[ijack][t]/njacks;
        }
        mass_err[t]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_mass_ave[t]-mass_ave[t]*mass_ave[t]));
    }
    
    //t-range for the fit
    int t_min = delta_tmin;
    int t_max = delta_tmax;
    
    vvd_t coord(vd_t(0.0,T/2),1);
    for(int j=0; j<T/2; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvd_t jeff_mass(vd_t(0.0,coord.size()),njacks);
    
    jeff_mass = polyfit(coord,1,mass_err,M_eff,t_min,t_max);
    
    vd_t eff_mass_tmp(0.0,njacks);
    
#pragma omp parallel for
    for(int ijack=0;ijack<njacks;ijack++)
        eff_mass_tmp[ijack]=jeff_mass[ijack][0];
    
    ofstream outfile;
    outfile.open(path_to_ens+"eff_mass_sea_array", ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            outfile.write((char*) &eff_mass_tmp[ijack],sizeof(double));
        outfile.close();
    }
    else cerr<<"Unable to create the output file \"eff_mass_sea_array\" "<<endl;
}

double per_two_pts_corr_with_ins_ratio_fun(const double &M,const double &TH,const double &t)
{return -(t-TH)*tanh(M*(t-TH));}

vd_t oper_t::effective_slope(vd_t data, vd_t M, int TH)
{
    int dt=1;
    vd_t out(0.0,data.size());
    
//    cout<<"DEBUG_____"<<endl;
//    cout<<"data size: "<<data.size()<<endl;
//    cout<<"M size: "<<M.size()<<endl;
//    cout<<"out size: "<<out.size()<<endl;

    
    for(size_t t=0;t<data.size()-dt;t++)
    {
        double num = data[t+dt]-data[t];
        double den = per_two_pts_corr_with_ins_ratio_fun(M[t],(double)TH,(double)(t+dt)) -
                     per_two_pts_corr_with_ins_ratio_fun(M[t],(double)TH,(double)(t));
        
        out[t]=num/den;
    }
    out[data.size()-dt]=0.0;
        
    return out;
}

