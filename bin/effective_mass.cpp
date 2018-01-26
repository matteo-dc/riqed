#include "global.hpp"
#include "aliases.hpp"
#include "contractions.hpp"
#include "fit.hpp"
#include <iostream>
#include <omp.h>

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

// compute effective mass
void compute_eff_mass()
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
            jP5P5_00[mr_fw][mr_bw]=get_contraction(mr_fw,"0",mr_bw,"0","P5P5","RE","EVEN",conf_id);
    
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
    
//    vvvvd_t eff_mass_fit_parameters(vvvd_t(vvd_t(vd_t(0.0,2),coord.size()),nmr),nmr);
    
    vvvvd_t jeff_mass(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nmr),nmr);
    
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            jeff_mass[m_fw][m_bw] = polyfit(coord, 1, mass_err[m_fw][m_bw], M_eff[m_fw][m_bw], t_min, t_max);
//        {
//            eff_mass_fit_parameters[mr_fw][mr_bw]=fit_par(coord,mass_err[mr_fw][mr_bw],M_eff[mr_fw][mr_bw],t_min,t_max);
//        }

//    vvvd_t eff_mass_array(vvd_t(vd_t(0.0,2),nmr),nmr);
    
//    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
//        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
//        {
//            eff_mass_array[mr_fw][mr_bw][0]=eff_mass_fit_parameters[mr_fw][mr_bw][0][0];
//            eff_mass_array[mr_fw][mr_bw][1]=eff_mass_fit_parameters[mr_fw][mr_bw][0][1];
//        }
    
    vvvd_t eff_mass(vvd_t(vd_t(0.0,nmr),nmr),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;m_fw++)
            for(int mr_bw=0;mr_bw<nmr;m_bw++)
                deltam_cr[ijack][mr_fw][mr_bw]=jdeltam_cr[mr_fw][mr_bw][ijack][0];
    
    
    ofstream outfile;
    outfile.open("eff_mass_array", ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                {
                    int r1 = mr_fw%nr;
                    int m1 = (mr_fw-r1)/nr;
                    int r2 = mr_bw%nr;
                    int m2 = (mr_bw-r2)/nr;
                    
                    for(int i=0;i<2;i++)
                        outfile.write((char*) &eff_mass_array[ijack][mr_fw][mr_bw][i],sizeof(double));
                }
        

        outfile.close();
    }
    else cerr<<"Unable to create the output file \"eff_mass_array\" "<<endl;
}
