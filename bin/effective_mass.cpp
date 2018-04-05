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
    int _nmr=0, _nm=0;
    if(suffix.compare("")==0)
    {
        _nm=nm;
        _nmr=nmr;
    }
    else if(suffix.compare("sea")==0)
    {
        _nm=1;
        _nmr=nr;
    }
    
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    // define jackknife V0P5 correlators
    vvvvd_t jP5P5_00(vvvd_t(vvd_t(vd_t(T/2+1),njacks),_nmr),_nmr);
    vvvvd_t jV0P5_00(vvvd_t(vvd_t(vd_t(T/2+1),njacks),_nmr),_nmr);
    
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

#pragma omp parallel for collapse(2)
    for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
        {
            jP5P5_00[mr_fw][mr_bw]=get_contraction(suffix,out_hadr,mr_fw,"0",mr_bw,"0","P5P5","RE","EVEN",conf_id,path_to_ens);
            jV0P5_00[mr_fw][mr_bw]=get_contraction(suffix,out_hadr,mr_fw,"0",mr_bw,"0","V0P5","IM","ODD",conf_id,path_to_ens);
        }
    
    //average over r
#pragma omp parallel for collapse(4)
    for(int m_fw=0;m_fw<_nm;m_fw++)
        for(int m_bw=0;m_bw<_nm;m_bw++)
            for (int ijack=0; ijack<njacks; ijack++)
                for(int t=0;t<T/2+1;t++)
                    for(int r=0;r<nr;r++)
                    {
                        int r1=r;
                        int r2=(r+1)%2;
                        
                        int cr_even = r2 + r1;
                        int cr_odd = r2 - r1;
                        
                        jV0P5_00_ave[m_fw][m_bw][ijack][t] += jV0P5_00[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_odd/nr;
                        jP5P5_00_ave[m_fw][m_bw][ijack][t] += jP5P5_00[r1+nr*m_fw][r1+nr*m_bw][ijack][t]*cr_even/nr;
                    }

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

    cout<<endl;
    ofstream outfile;
    outfile.open(path_to_ens+"mPCAC", ios::out | ios::binary);
    
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
    else cerr<<"Unable to create the output file \"mPCAC\" "<<endl;
}

// compute effective mass
void oper_t::compute_eff_mass()
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    // define jackknife V0P5 correlators
    vvvvd_t jP5P5_00(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nmr),nmr);
    
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            jP5P5_00[mr_fw][mr_bw]=get_contraction("",out_hadr,mr_fw,"0",mr_bw,"0","P5P5","RE","EVEN",conf_id,path_to_ens);
    
    //   cout<<"**********DEBUG: P5P5 correlator  *************"<<endl;
    //   for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    //     for(int mr_bw=0;mr_bw<nmr;mr_bw++)
    //       for(int ijack=0;ijack<njacks;ijack++)
    //   	for(int t=0;t<T/2-1;t++)
    //   	  cout<<mr_fw<<" "<<mr_bw<<" ijack "<<ijack<<" t "<<t<<"\t"<< jP5P5_00[mr_fw][mr_bw][ijack][t]/jP5P5_00[mr_fw][mr_bw][ijack][t+1]<<endl;
    //   cout<<"**********DEBUG********************************"<<endl;
    
    // define effective mass array
    vvvvd_t M_eff(vvvd_t(vvd_t(vd_t(T/2),njacks),nmr),nmr);
    
#pragma omp parallel for collapse(4)
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            for(int ijack=0; ijack<njacks;ijack++)
                for(int t=0;t<T/2;t++)
                    M_eff[mr_fw][mr_bw][ijack][t] = solve_Newton (jP5P5_00[mr_fw][mr_bw],ijack,t,T);
    
    //  cout<<"**********DEBUG*************"<<endl;
    //  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    //    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
    //      for(int ijack=0;ijack<njacks;ijack++)
    //  	for(int t=0;t<T/2;t++)
    //  	  cout<<mr_fw<<" "<<mr_bw<<" ijack "<<ijack<<" t "<<t<<"\t"<<M_eff[mr_fw][mr_bw][ijack][t]<<endl;
    //  cout<<"**********DEBUG*************"<<endl;
    
    // cout<<"**********DEBUG*************"<<endl;
    // for(double i=0;i<10;i+=0.1){ cout<<i+1<<"\t"<<f_mass(22,T,i,jP5P5_00[0][0][0][22]/jP5P5_00[0][0][0][23])<<endl;}
    // cout<<"**********DEBUG*************"<<endl;
    
    
    vvvd_t mass_ave(vvd_t(vd_t(0.0,T/2),nmr),nmr), sqr_mass_ave(vvd_t(vd_t(0.0,T/2),nmr),nmr), mass_err(vvd_t(vd_t(0.0,T/2),nmr),nmr);
    
#pragma omp parallel for collapse(3)
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            for(int t=0;t<T/2;t++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mass_ave[mr_fw][mr_bw][t]+=M_eff[mr_fw][mr_bw][ijack][t]/njacks;
                    sqr_mass_ave[mr_fw][mr_bw][t]+=M_eff[mr_fw][mr_bw][ijack][t]*M_eff[mr_fw][mr_bw][ijack][t]/njacks;
                }
                
                mass_err[mr_fw][mr_bw][t]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_mass_ave[mr_fw][mr_bw][t]-mass_ave[mr_fw][mr_bw][t]*mass_ave[mr_fw][mr_bw][t]));
                
                //	  cout<<"**********DEBUG*************"<<endl;
                //	  cout<<mr_fw<<" "<<mr_bw<<" "<<" t  "<<t<<"  "<<mass_ave[mr_fw][mr_bw][t]<<"  +/-  "<< mass_err[mr_fw][mr_bw][t] <<endl;

            }
    
    //t-range for the fit
    int t_min = delta_tmin;
    int t_max = delta_tmax;
    
    vvd_t coord(vd_t(0.0,T/2),1);
    for(int j=0; j<T/2; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvd_t jeff_mass(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nmr),nmr);
    
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            jeff_mass[mr_fw][mr_bw] = polyfit(coord, 1, mass_err[mr_fw][mr_bw], M_eff[mr_fw][mr_bw], t_min, t_max);
    
    vvvd_t eff_mass_tmp(vvd_t(vd_t(0.0,nmr),nmr),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                eff_mass_tmp[ijack][mr_fw][mr_bw]=jeff_mass[mr_fw][mr_bw][ijack][0];
    
    
    ofstream outfile;
    outfile.open(path_to_ens+"eff_mass_array", ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                {
                    outfile.write((char*) &eff_mass_tmp[ijack][mr_fw][mr_bw],sizeof(double));
                }

        outfile.close();
    }
    else cerr<<"Unable to create the output file \"eff_mass_array\" "<<endl;
}

// compute effective sea mass
void oper_t::compute_eff_mass_sea()
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    int T=size[0];
    
    // define jackknife V0P5 correlators
    vvvvd_t jP5P5_00(vvvd_t(vvd_t(vd_t(T/2+1),njacks),nr),nr);
    
    for(int r1=0; r1<nr; r1++)
      for(int r2=0; r2<nr; r2++)
	jP5P5_00[r1][r2]=get_contraction("sea",out_hadr,r1,"0",r2,"0","P5P5","RE","EVEN",conf_id,path_to_ens);
	  
    //   cout<<"**********DEBUG: P5P5 correlator  *************"<<endl;
    //   for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    //     for(int mr_bw=0;mr_bw<nmr;mr_bw++)
    //       for(int ijack=0;ijack<njacks;ijack++)
    //   	for(int t=0;t<T/2-1;t++)
    //   	  cout<<mr_fw<<" "<<mr_bw<<" ijack "<<ijack<<" t "<<t<<"\t"<< jP5P5_00[mr_fw][mr_bw][ijack][t]/jP5P5_00[mr_fw][mr_bw][ijack][t+1]<<endl;
    //   cout<<"**********DEBUG********************************"<<endl;
    
    // define effective mass array
    vvvvd_t M_eff(vvvd_t(vvd_t(vd_t(T/2),njacks),nr),nr);
    
#pragma omp parallel for collapse(4)
    for(int r1=0;r1<nr;r1++)
        for(int r2=0;r2<nr;r2++)
            for(int ijack=0; ijack<njacks;ijack++)
                for(int t=0;t<T/2;t++)
                    M_eff[r1][r2][ijack][t] = solve_Newton (jP5P5_00[r1][r2],ijack,t,T);
    
    //  cout<<"**********DEBUG*************"<<endl;
    //  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    //    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
    //      for(int ijack=0;ijack<njacks;ijack++)
    //  	for(int t=0;t<T/2;t++)
    //  	  cout<<mr_fw<<" "<<mr_bw<<" ijack "<<ijack<<" t "<<t<<"\t"<<M_eff[mr_fw][mr_bw][ijack][t]<<endl;
    //  cout<<"**********DEBUG*************"<<endl;
    
    // cout<<"**********DEBUG*************"<<endl;
    // for(double i=0;i<10;i+=0.1){ cout<<i+1<<"\t"<<f_mass(22,T,i,jP5P5_00[0][0][0][22]/jP5P5_00[0][0][0][23])<<endl;}
    // cout<<"**********DEBUG*************"<<endl;
    
    
    vvvd_t mass_ave(vvd_t(vd_t(0.0,T/2),nr),nr), sqr_mass_ave(vvd_t(vd_t(0.0,T/2),nr),nr), mass_err(vvd_t(vd_t(0.0,T/2),nr),nr);
    
#pragma omp parallel for collapse(3)
    for(int r1=0;r1<nr;r1++)
        for(int r2=0;r2<nr;r2++)
            for(int t=0;t<T/2;t++)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    mass_ave[r1][r2][t]+=M_eff[r1][r2][ijack][t]/njacks;
                    sqr_mass_ave[r1][r2][t]+=M_eff[r1][r2][ijack][t]*M_eff[r1][r2][ijack][t]/njacks;
                }
                
                mass_err[r1][r2][t]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_mass_ave[r1][r2][t]-mass_ave[r1][r2][t]*mass_ave[r1][r2][t]));
                
                //	  cout<<"**********DEBUG*************"<<endl;
                //	  cout<<mr_fw<<" "<<mr_bw<<" "<<" t  "<<t<<"  "<<mass_ave[mr_fw][mr_bw][t]<<"  +/-  "<< mass_err[mr_fw][mr_bw][t] <<endl;
                
            }
    
    //t-range for the fit
    int t_min = delta_tmin;
    int t_max = delta_tmax;
    
    vvd_t coord(vd_t(0.0,T/2),1);
    for(int j=0; j<T/2; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    vvvvd_t jeff_mass(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nr),nr);
    
    for(int r1=0;r1<nr;r1++)
        for(int r2=0;r2<nr;r2++)
            jeff_mass[r1][r2] = polyfit(coord, 1, mass_err[r1][r2], M_eff[r1][r2], t_min, t_max);
    
    vvvd_t eff_mass_tmp(vvd_t(vd_t(0.0,nr),nr),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int r1=0;r1<nr;r1++)
            for(int r2=0;r2<nr;r2++)
                eff_mass_tmp[ijack][r1][r2]=jeff_mass[r1][r2][ijack][0];
    
    
    ofstream outfile;
    outfile.open(path_to_ens+"eff_mass_sea_array", ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int r1=0;r1<nr;r1++)
                for(int r2=0;r2<nr;r2++)
                {
                    outfile.write((char*) &eff_mass_tmp[ijack][r1][r2],sizeof(double));
                }
        
        outfile.close();
    }
    else cerr<<"Unable to create the output file \"eff_mass_sea_array\" "<<endl;
}



