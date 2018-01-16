#include "global.hpp"
#include "aliases.hpp"
#include "contractions.hpp"
#include "jack.hpp"
#include "fit.hpp"
#include <iostream>


//#ifdef HAVE_CONFIG_H
//#include <config.hpp>
//#endif
//
//#include <complex>
//#include <fstream>
//#include <iostream>
//#include <vector>
//#include <Eigen/Dense>
//#include <valarray>
//#include <math.h>
//#include <chrono>
//
//using namespace std;
//using namespace Eigen;
//using namespace std::chrono;
//
//
////coordinates in the lattice
//using coords_t=array<int,4>;
//
////complex double
//using dcompl=complex<double>;
//
////propagator (12X12)
//using prop_t=Matrix<dcompl,12,12>;
//
////list of propagators
//using vprop_t=valarray<prop_t>;
//using vvprop_t=valarray< valarray<prop_t> >;
//using vvvprop_t=valarray< valarray< valarray<prop_t> > >;
//
////list of gamma for a given momentum
//using qline_t=valarray<prop_t>;
//using vqline_t=valarray<qline_t>;
//using vvqline_t=valarray<vqline_t>;
//using vert_t = vvqline_t;
//
////list of jackknife propagators
//using jprop_t=valarray< valarray<prop_t> >;
//
////list of jackknife vertices
//using jvert_t=valarray< vert_t >;
//
////valarray of complex double
//using vd_t=valarray<double>;
//
////valarray of valarray of complex double
//using vvd_t=valarray< valarray<double> > ;
//
////valarray of valarray of valarray of complex double
//using vvvd_t=valarray< valarray< valarray<double> > >;
//using vvvvd_t=valarray<vvvd_t>;
//
////valarray of complex double
//using vdcompl_t=valarray<dcompl>;
//using vvdcompl_t=valarray< vdcompl_t >;
//using vvvdcompl_t=valarray< vvdcompl_t >;
//using vvvvdcompl_t=valarray< vvvdcompl_t >;
//
////useful notation
//using jZ_t=vvd_t;
//using jZbil_t=vvvvd_t;
//using jproj_t=vvvvd_t;
//
////valarray of Eigen Vectors
//using vXd_t=valarray<VectorXd>;
//using vvXd_t=valarray<vXd_t>;
//using vvvXd_t=valarray<vvXd_t>;
//
//
////list of momenta
//vector<coords_t> mom_list;
//
////list of N(p)
//vector<int> Np;
//
//int nr,nm,nmr;

////create the path-string to the contraction
//string path_to_contr(const string &string_path, int i_conf,const int mr1, const string &T1, const int mr2, const string &T2)
//{
//    
//    int r1 = mr1%nr;
//    int m1 = (mr1-r1)/nr;
//    int r2 = mr2%nr;
//    int m2 = (mr2-r2)/nr;
//    
//    char path[1024];
//    sprintf(path,"%sout/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",string_path.c_str(),i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());
//    
//    // cout<<path<<endl;
//    
//    return path;
//}
//
//
////jackknife Propagator
//vvd_t jackknife_double(vvd_t &jd, int size, int nconf, int clust_size )
//{
//    vd_t jSum(0.0,size);
//    
//    //sum of jd
//    for(size_t j=0;j<jd.size();j++) jSum+= jd[j];
//    //jackknife fluctuation
//    for(size_t j=0;j<jd.size();j++)
//    {
//        jd[j]=jSum-jd[j];
//        for(auto &it : jd[j])
//            it/=(nconf-clust_size);
//    }
//    
//    return jd;
//}


////compute fit parameters for a generic function f(x)=A+B*x+C*y(x)+D*z(x)+...
//vvd_t fit_par(const vvd_t &coord, const vd_t &error, const vvd_t &y, const int range_min, const int range_max,const string &path=NULL)
//{
//    int n_par = coord.size();
//    int njacks = y.size();
//    
//    MatrixXd S(n_par,n_par);
//    valarray<VectorXd> Sy(VectorXd(n_par),njacks);
//    valarray<VectorXd> jpars(VectorXd(n_par),njacks);
//    
//    //initialization
//    S=MatrixXd::Zero(n_par,n_par);
//    for(int ijack=0; ijack<njacks; ijack++)
//    {
//        Sy[ijack]=VectorXd::Zero(n_par);
//        jpars[ijack]=VectorXd::Zero(n_par);
//    }
//    
//    //definition
//    for(int i=range_min; i<=range_max; i++)
//    {
//        for(int j=0; j<n_par; j++)
//            for(int k=0; k<n_par; k++)
//                if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
//        
//        for(int ijack=0; ijack<njacks; ijack++)
//            for(int k=0; k<n_par; k++)
//                if(std::isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]);
//    }
//    
//    for(int ijack=0; ijack<njacks; ijack++)
//        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
//    
//    vvd_t par_array(vd_t(0.0,2),n_par);
//    
//    vd_t par_ave(0.0,n_par), par2_ave(0.0,n_par), par_err(0.0,n_par);
//    
//    for(int k=0; k<n_par; k++)
//    {
//        for(int ijack=0;ijack<njacks;ijack++)
//        {
//            par_ave[k]+=jpars[ijack](k)/njacks;
//            par2_ave[k]+=jpars[ijack](k)*jpars[ijack](k)/njacks;
//        }
//        par_err[k]=sqrt((double)(njacks-1))*sqrt(fabs(par2_ave[k]-par_ave[k]*par_ave[k]));
//        
//        par_array[k][0] = par_ave[k];
//        par_array[k][1] = par_err[k];
//    }
//
////    if(path!="")
////    {
////        ofstream out(path);
////        out<<"@type xydy"<<endl;
////        for(int i=1; i<range_max; i++)
////            out<<i<<" "<<y[0][i]<<" "<<error[i]<<endl;
////        out<<"&"<<endl;
////        out<<"@type xy"<<endl;
////        out<<range_min<<" "<<par_ave[0]-par_err[0]<<endl;
////        out<<range_min<<" "<<par_ave[0]+par_err[0]<<endl;
////        out<<range_max<<" "<<par_ave[0]+par_err[0]<<endl;
////        out<<range_min<<" "<<par_ave[0]-par_err[0]<<endl;
////        out<<range_max<<" "<<par_ave[0]-par_err[0]<<endl;
////    }
//    return par_array;
//    
//}

//valarray<VectorXd> fit_par_jackknife(const vvd_t &coord, vd_t &error, const vvd_t &y, const int range_min, const int range_max)
//{
//    int n_par = coord.size();
//    int njacks = y.size();
//    
//    MatrixXd S(n_par,n_par);
//    valarray<VectorXd> Sy(VectorXd(n_par),njacks);
//    valarray<VectorXd> jpars(VectorXd(n_par),njacks);
//    
//    //initialization
//    S=MatrixXd::Zero(n_par,n_par);
//    for(int ijack=0; ijack<njacks; ijack++)
//    {
//        Sy[ijack]=VectorXd::Zero(n_par);
//        jpars[ijack]=VectorXd::Zero(n_par);
//    }
//    
//    //definition
//    for(int i=range_min; i<=range_max; i++)
//    {
//        if(error[i]<1.0e-20) error[i]+=1.0e-20;
//        
//        //cout<<y[0][i]<<"\t"<<error[i]<<endl;
//        
//        for(int j=0; j<n_par; j++)
//            for(int k=0; k<n_par; k++)
//                if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
//        
//        for(int ijack=0; ijack<njacks; ijack++)
//            for(int k=0; k<n_par; k++)
//                if(std::isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]);
//    }
//    
//    // cout<<endl;
//    
//    for(int ijack=0; ijack<njacks; ijack++)
//        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
//    
//    for(int i=range_min; i<=range_max; i++)
//        cout<<"(x,y) [ijack=0] = "<<coord[0][i]<<" "<<y[0][i]<<" "<<error[i]<<endl;
//    cout<<"Extrapolation: "<<jpars[0](0)<<endl;
//    
//    return jpars;
//    
//}


//vvd_t get_contraction(const int mr1, const string &T1, const int mr2, const string &T2, const string &ID, const string &reim, const string &parity, const int T, const int nconfs, const int njacks , const int* conf_id, const string &string_path)
//{
//    
//    vd_t data_V0P5_real(0.0,T);
//    vd_t data_V0P5_imag(0.0,T);
//    vd_t data_P5P5_real(0.0,T);
//    vd_t data_P5P5_imag(0.0,T);
//    
//    vvd_t jP5P5_real(vd_t(0.0,T),njacks);
//    vvd_t jP5P5_imag(vd_t(0.0,T),njacks);
//    vvd_t jV0P5_real(vd_t(0.0,T),njacks);
//    vvd_t jV0P5_imag(vd_t(0.0,T),njacks);
//    
//    int clust_size=nconfs/njacks;
//    
//    /////////
//    
//    for(int iconf=0;iconf<nconfs;iconf++)
//    {
//        int ijack=iconf/clust_size;
//        
//        ifstream infile;
//        
//        string path=path_to_contr(string_path,conf_id[iconf],mr1,T1,mr2,T2);
//        cout<<"opening: "<<path<<endl;
//        infile.open(path);
//        
//        if(!infile.good())
//        {cerr<<"Unable to open file "<<path_to_contr(string_path,conf_id[iconf],mr1,T1,mr2,T2)<<endl;
//            exit(1);}
//        
//        //DEBUG
//        // cout<<"  Reading contraction from "<<path_to_contr(conf_id[iconf],mr1,T1,mr2,T2)<<endl;
//        //DEBUG
//        
//        infile.ignore(256,'5');
//        
//        for(int t=0; t<T; t++)
//        {
//            infile>>data_V0P5_real[t];
//            infile>>data_V0P5_imag[t];
//        }
//        
//        infile.ignore(256,'5');
//        infile.ignore(256,'5');
//        
//        for(int t=0; t<T; t++)
//        {
//            infile>>data_P5P5_real[t];
//            infile>>data_P5P5_imag[t];
//        }
//        
//        for(int t=0; t<T; t++) jV0P5_real[ijack][t]+=data_V0P5_real[t];
//        for(int t=0; t<T; t++) jV0P5_imag[ijack][t]+=data_V0P5_imag[t];
//        for(int t=0; t<T; t++) jP5P5_real[ijack][t]+=data_P5P5_real[t];
//        for(int t=0; t<T; t++) jP5P5_imag[ijack][t]+=data_P5P5_imag[t];
//        
//        infile.close();
//    }
//    
//    jV0P5_real=jackknife_double(jV0P5_real,T,nconfs,clust_size);
//    jV0P5_imag=jackknife_double(jV0P5_imag,T,nconfs,clust_size);
//    jP5P5_real=jackknife_double(jP5P5_real,T,nconfs,clust_size);
//    jP5P5_imag=jackknife_double(jP5P5_imag,T,nconfs,clust_size);
//    
//    vvd_t jvec(vd_t(0.0,T),njacks);
//    
//    if(ID=="P5P5" and reim=="RE") jvec=jP5P5_real;
//    if(ID=="P5P5" and reim=="IM") jvec=jP5P5_imag;
//    if(ID=="V0P5" and reim=="RE") jvec=jV0P5_real;
//    if(ID=="V0P5" and reim=="IM") jvec=jV0P5_imag;
//    
//    double par;
//    
//    if(parity=="EVEN") par=1.0;
//    if(parity=="ODD") par=-1.0;
//    
//    vvd_t jvec_sym(vd_t(0.0,T),njacks);
//    vvd_t jvec_par(vd_t(0.0,T/2+1),njacks);
//    
//    for(int ijack=0;ijack<njacks;ijack++)
//    {
//        for(int t=0;t<T;t++)
//            jvec_sym[ijack][(T-t)%T]=jvec[ijack][t];
//        for(int t=0;t<T/2+1;t++)
//            jvec_par[ijack][t]=(jvec[ijack][t]+par*jvec_sym[ijack][t])/2.0;
//    }
//    
//    string path=ID+"_"+reim+"_mrbw_"+to_string(mr1)+"_mrfw_"+to_string(mr2)+"_"+T1+T2+".xmg";
//    ofstream out(path);
//    out<<"@type xy"<<endl;
//    for(int t=0;t<T;t++)
//        out<<t<<" "<<jvec_sym[0][t]<<endl;
//    
//    // if(ID=="P5P5" and reim=="RE" and parity=="EVEN"){
//    //   cout<<"**********DEBUG*************"<<endl;
//    //   for(int ijack=0;ijack<njacks;ijack++)
//    //     for(int t=0;t<T;t++)
//    // 	cout<<jvec[ijack][t]<<endl;
//    //   cout<<"**********DEBUG*************"<<endl;}
//    
//    return jvec_par;
//    
//}

//compute delta m_cr
/*vvvd_t*/void compute_deltam_cr(/*const int T, const int nconfs, const int njacks, const int* conf_id, const string &string_path, const int t_min, const int t_max*/)
{
//    int nmr=8;
//    int nr=2;
//    int nm=4;
    
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
            jV0P5_LL[mr_fw][mr_bw]=get_contraction(mr_bw,"F",mr_fw,"F","V0P5","IM","ODD",conf_id/*,T,nconfs,njacks,conf_id,string_path*/);
            jV0P5_0M[mr_fw][mr_bw]=get_contraction(mr_bw,"0",mr_fw,"FF","V0P5","IM","ODD",conf_id);
            jV0P5_M0[mr_fw][mr_bw]=get_contraction(mr_bw,"FF",mr_fw,"0","V0P5","IM","ODD",conf_id);
            jV0P5_0T[mr_fw][mr_bw]=get_contraction(mr_bw,"0",mr_fw,"T","V0P5","IM","ODD",conf_id);
            jV0P5_T0[mr_fw][mr_bw]=get_contraction(mr_bw,"T",mr_fw,"0","V0P5","IM","ODD",conf_id);
            //load the derivative wrt counterterm
            jV0P5_0P[mr_fw][mr_bw]=get_contraction(mr_bw,"0",mr_fw,"P","V0P5","RE","ODD",conf_id);
            jV0P5_P0[mr_fw][mr_bw]=get_contraction(mr_bw,"P",mr_fw,"0","V0P5","RE","ODD",conf_id);
        }
    
    //average over r
//#pragma omp parallel for collapse(5)
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int r=0;r<nr;r++)
                for (int ijack=0; ijack<njacks; ijack++)
                    for(int t=0;t<T/2+1;t++)
                    {
                        int cr_even = (r+1)%2 + (r%2);
                        int cr_odd = (r+1)%2 - (r%2);
                        
                        jV0P5_LL_ave[m_fw][m_bw][ijack][t] += jV0P5_LL[r+nr*m_fw][r+nr*m_bw][ijack][t]*cr_odd/nr;
                        jV0P5_0M_ave[m_fw][m_bw][ijack][t] += jV0P5_0M[r+nr*m_fw][r+nr*m_bw][ijack][t]*cr_odd/nr;
                        jV0P5_M0_ave[m_fw][m_bw][ijack][t] += jV0P5_M0[r+nr*m_fw][r+nr*m_bw][ijack][t]*cr_odd/nr;
                        jV0P5_0T_ave[m_fw][m_bw][ijack][t] += jV0P5_0T[r+nr*m_fw][r+nr*m_bw][ijack][t]*cr_odd/nr;
                        jV0P5_T0_ave[m_fw][m_bw][ijack][t] += jV0P5_T0[r+nr*m_fw][r+nr*m_bw][ijack][t]*cr_odd/nr;
                        jV0P5_0P_ave[m_fw][m_bw][ijack][t] += jV0P5_0P[r+nr*m_fw][r+nr*m_bw][ijack][t]*cr_even/nr;
                        jV0P5_P0_ave[m_fw][m_bw][ijack][t] += jV0P5_P0[r+nr*m_fw][r+nr*m_bw][ijack][t]*cr_even/nr;
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
                
//                    printf("m_fw: %d m_bw: %d ijack: %d t: %d delta: %lf \n",m_fw,m_bw,ijack,t,deltam_cr_corr[m_fw][m_bw][ijack][t]);
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
    
    //t-range for the fit
    // int t_min=12;
    // int t_max=23;
    
    vvd_t coord(vd_t(0.0,T/2+1),1);
    for(int j=0; j<T/2+1; j++)
    {
        coord[0][j] = 1.0;  //fit a costante
    }
    
    // vvvvd_t deltam_cr_fit_parameters(vvvd_t(vvd_t(vd_t(0.0,2),coord.size()),nmr),nmr);
    
    //  for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    //    for(int mr_bw=0;mr_bw<nmr;mr_bw++)
    //      deltam_cr_fit_parameters[mr_fw][mr_bw]=fit_par(coord,error[mr_fw][mr_bw],deltam_cr_corr[mr_fw][mr_bw],t_min,t_max,"plot_deltam_cr_mrfw_"+to_string(mr_fw)+"_mrbw_"+to_string(mr_bw));
    //
    
//    vvvXd_t jdeltam_cr(vvXd_t(vXd_t(VectorXd(coord.size()),njacks),nm),nm);
    vvvvd_t jdeltam_cr(vvvd_t(vvd_t(vd_t(0.0,coord.size()),njacks),nm),nm);
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            jdeltam_cr[m_fw][m_bw] = fit_par_jackknife(coord, 1, error[m_fw][m_bw], deltam_cr_corr[m_fw][m_bw], delta_tmin, delta_tmax);
    
    //    vvvd_t deltam_cr(vvd_t(vd_t(0.0,2),nmr),nmr);
    //    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
    //        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
    //        {
    //            deltam_cr[mr_fw][mr_bw][0]=deltam_cr_fit_parameters[mr_fw][mr_bw][0][0];
    //            deltam_cr[mr_fw][mr_bw][1]=deltam_cr_fit_parameters[mr_fw][mr_bw][0][1];
    //        }
    
    vvvd_t deltam_cr(vvd_t(vd_t(0.0,nm),nm),njacks);
    
#pragma omp parallel for collapse(3)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
                deltam_cr[ijack][m_fw][m_bw]=jdeltam_cr[m_fw][m_bw][ijack][0];
    
//    return deltam_cr;
//}
// 
//
//void compute_and_print_deltam_cr()
//{
//int main(int narg,char **arg)
//{
//    
//    if (narg!=6){
//        cerr<<"Number of arguments not valid:  <nconfs> <njacks> <tmin> <tmax> <path before 'out' directory: /marconi_work/.../ >"<<endl;
//        exit(0);
//    }
//    
//    int nconfs=stoi(arg[1]);
//    int njacks=stoi(arg[2]);
//    //int clust_size=nconfs/njacks;
////    int conf_id[nconfs];
//    double T=48;//,L=24
//    //size_t nhits=1; //!
//    
//    int tmin = stoi(arg[3]);
//    int tmax = stoi(arg[4]);
//    
//    string string_path = arg[5];
//    
//    nm = 4;  //! to be passed from command line
//    nr = 2;
//    
    //nmr=nm*nr;
    
//    for(int iconf=0;iconf<nconfs;iconf++)
//        conf_id[iconf]=100+iconf*1;
    
//    vvvd_t deltam_cr_array = compute_deltam_cr(/*T,nconfs,njacks,conf_id,string_path,tmin,tmax*/); //deltam_cr_array[ijack][m_fw][m_bw]
    
    ofstream outfile;
    outfile.open("deltam_cr_array", ios::out | ios::binary);
    
    if (outfile.is_open())
    {
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m_fw=0;m_fw<nm;m_fw++)
                for(int m_bw=0;m_bw<nm;m_bw++)
                {
                    outfile.write((char*) &deltam_cr/*_array*/[ijack][m_fw][m_bw],sizeof(double));
               
//                    cout<<"ijack "<<ijack<<" m1 "<<m_fw<<" m2 "<<m_bw<<" delta: "<<deltam_cr/*_array*/[ijack][m_fw][m_bw]<<endl;
                }
        
        outfile.close();
    }
    else cerr<< "Unable to create the output file \"deltam_cr_array\" "<<endl;
    
//    return 0;
    
}
