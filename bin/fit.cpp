#include "aliases.hpp"
#include "global.hpp"
#include <iostream>

vvd_t polyfit(const vvd_t &coord, const int n_par, vd_t &error, const vvd_t &y, const double xmin, const double xmax)
{
    //    int njacks = y.size();
    
    MatrixXd S(n_par,n_par);
    vXd_t Sy(VectorXd(n_par),njacks);
    vXd_t jpars(VectorXd(n_par),njacks);
    
    vvd_t jvpars(vd_t(0.0,n_par),njacks);

    int xsize=coord[0].size();
    
    //initialization
    S=MatrixXd::Zero(n_par,n_par);
    for(int ijack=0; ijack<njacks; ijack++)
    {
        Sy[ijack]=VectorXd::Zero(n_par);
        jpars[ijack]=VectorXd::Zero(n_par);
    }
    
    //definition
    for(int i=0; i<xsize; i++)
    {
        if(coord[1][i]>=xmin and coord[1][i]<=xmax)
        {
            if(error[i]<1.0e-20) error[i]+=1.0e-20;
            
            for(int j=0; j<n_par; j++)
                for(int k=0; k<n_par; k++)
                    if(std::isnan(error[i])==0)
                        S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
            
            for(int ijack=0; ijack<njacks; ijack++)
                for(int k=0; k<n_par; k++)
                    if(std::isnan(error[i])==0)
                        Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]);
        }
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
    {
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
        
        for(int ipar=0;ipar<n_par;ipar++) jvpars[ijack][ipar]=jpars[ijack](ipar);
    }
    
    //    for(int i=range_min; i<=range_max; i++)
    //        cout<<"(x,y) [ijack=0] = "<<coord[0][i]<<" "<<y[0][i]<<" "<<error[i]<<endl;
    //    cout<<"Extrapolation (jpars): "<<jpars[0](0)<<endl;
    //    cout<<"Extrapolation (jvpars): "<<jvpars[0][0]<<endl;
    
    return jvpars;
    
}

vvd_t polyfit(const vvd_t &coord, const int n_par, vd_t &error, const vvd_t &y, const int range_min, const int range_max)
{
//    int njacks = y.size();
    
    MatrixXd S(n_par,n_par);
    vXd_t Sy(VectorXd(n_par),njacks);
    vXd_t jpars(VectorXd(n_par),njacks);
    
    vvd_t jvpars(vd_t(0.0,n_par),njacks);
    
    //initialization
    S=MatrixXd::Zero(n_par,n_par);
    for(int ijack=0; ijack<njacks; ijack++)
    {
        Sy[ijack]=VectorXd::Zero(n_par);
        jpars[ijack]=VectorXd::Zero(n_par);
    }
    
    //definition
    for(int i=range_min; i<=range_max; i++)
    {
        if(error[i]<1.0e-20) error[i]+=1.0e-20;
        
        for(int j=0; j<n_par; j++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
        
        for(int ijack=0; ijack<njacks; ijack++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]);
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
    {
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
        
        for(int ipar=0;ipar<n_par;ipar++) jvpars[ijack][ipar]=jpars[ijack](ipar);
    }
    
//    for(int i=range_min; i<=range_max; i++)
//        cout<<"(x,y) [ijack=0] = "<<coord[0][i]<<" "<<y[0][i]<<" "<<error[i]<<endl;
//    cout<<"Extrapolation (jpars): "<<jpars[0](0)<<endl;
//    cout<<"Extrapolation (jvpars): "<<jvpars[0][0]<<endl;
    
    return jvpars;
    
}

vd_t fit_continuum(const vvd_t &coord, vd_t &error, const vd_t &y, const int p2_min, const int p2_max, const double &p_min_value)
{
    int n_par = coord.size();
    //int nbil = y[0].size();
    
    MatrixXd S(n_par,n_par);
    VectorXd Sy(n_par);
    VectorXd jpars(n_par);
    vd_t jvpars(0.0,n_par);
    
    //initialization
    S=MatrixXd::Zero(n_par,n_par);
    Sy=VectorXd::Zero(n_par);
    jpars=VectorXd::Zero(n_par);
    
    //definition
    for(int i=p2_min; i<p2_max; i++)
    {
        if(error[i]<1e-50)
            error[i]+=1e-50;
        
        if(coord[1][i]>p_min_value)
        {
            for(int j=0; j<n_par; j++)
                for(int k=0; k<n_par; k++)
                    if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
            
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) Sy(k) += y[i]*coord[k][i]/(error[i]*error[i]);
        }
    }
    
    jpars = S.colPivHouseholderQr().solve(Sy);
    
    for(int ipar=0;ipar<n_par;ipar++) jvpars[ipar]=jpars(ipar);
    
    return jvpars; //jpars[ibil][ijack][ipar]
}



vXd_t fit_chiral_jackknife(const vvd_t &coord, vd_t &error, const vector<vd_t> &y, const int range_min, const int range_max, const double &p_min_value)
{
    int n_par = coord.size();
    
    MatrixXd S(n_par,n_par);
    vXd_t Sy(VectorXd(n_par),njacks);
    vXd_t jpars(VectorXd(n_par),njacks);
    
    //initialization
    S=MatrixXd::Zero(n_par,n_par);
    for(int ijack=0; ijack<njacks; ijack++)
    {
        Sy[ijack]=VectorXd::Zero(n_par);
        jpars[ijack]=VectorXd::Zero(n_par);
    }
    
    //definition
    for(int i=range_min; i<range_max; i++)
    {
        if(error[i]<1e-50) error[i]+=1e-50;
        
        if(coord[1][i]>p_min_value)
        {
            for(int j=0; j<n_par; j++)
                for(int k=0; k<n_par; k++)
                    if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
            
            for(int ijack=0; ijack<njacks; ijack++)
                for(int k=0; k<n_par; k++)
                    if(std::isnan(error[i])==0) Sy[ijack](k) += y[i][ijack]*coord[k][i]/(error[i]*error[i]);
        }
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
    
    return jpars;
    
}

vvXd_t fit_chiral_jackknife(const vvd_t &coord, vvd_t &error, const vector<vvd_t> &y, const int range_min, const int range_max, const double &p_min_value)
{
    int n_par = coord.size();
    //int nbil = y[0].size();
    
    valarray<MatrixXd> S(MatrixXd(n_par,n_par),nbil);
    vvXd_t Sy(valarray<VectorXd>(VectorXd(n_par),njacks),nbil);
    vvXd_t jpars(valarray<VectorXd>(VectorXd(n_par),njacks),nbil);
    
    //initialization
    for(int ibil=0; ibil<nbil;ibil++)
        S[ibil]=MatrixXd::Zero(n_par,n_par);
    
    for(int ibil=0; ibil<nbil;ibil++)
        for(int ijack=0; ijack<njacks; ijack++)
        {
            Sy[ibil][ijack]=VectorXd::Zero(n_par);
            jpars[ibil][ijack]=VectorXd::Zero(n_par);
        }
    
    //definition
    for(int i=range_min; i<range_max; i++)
    {
        for(int ibil=0; ibil<nbil;ibil++)
        {
            if(error[i][ibil]<1e-50)
                error[i][ibil]+=1e-50;
        }
        
        if(coord[1][i]>p_min_value)
        {
            for(int ibil=0; ibil<nbil;ibil++)
                for(int j=0; j<n_par; j++)
                    for(int k=0; k<n_par; k++)
                        if(std::isnan(error[i][ibil])==0) S[ibil](j,k) += coord[j][i]*coord[k][i]/(error[i][ibil]*error[i][ibil]);
            
            for(int ibil=0; ibil<nbil;ibil++)
                for(int ijack=0; ijack<njacks; ijack++)
                    for(int k=0; k<n_par; k++)
                        if(std::isnan(error[i][ibil])==0) Sy[ibil][ijack](k) += y[i][ibil][ijack]*coord[k][i]/(error[i][ibil]*error[i][ibil]);
        }
    }
    
    for(int ibil=0; ibil<nbil;ibil++)
        for(int ijack=0; ijack<njacks; ijack++)
            jpars[ibil][ijack] = S[ibil].colPivHouseholderQr().solve(Sy[ibil][ijack]);
    
    return jpars; //jpars[ibil][ijack][ipar]
}

//compute fit parameters not jackknife
vvd_t fit_par(const vvd_t &coord, const vd_t &error, const vvd_t &y, const int range_min, const int range_max/*,const string &path=NULL*/)
{
    int n_par = coord.size();
    int njacks = y.size();
    
    MatrixXd S(n_par,n_par);
    valarray<VectorXd> Sy(VectorXd(n_par),njacks);
    valarray<VectorXd> jpars(VectorXd(n_par),njacks);
    
    //initialization
    S=MatrixXd::Zero(n_par,n_par);
    for(int ijack=0; ijack<njacks; ijack++)
    {
        Sy[ijack]=VectorXd::Zero(n_par);
        jpars[ijack]=VectorXd::Zero(n_par);
    }
    
    //definition
    for(int i=range_min; i<=range_max; i++)
    {
        for(int j=0; j<n_par; j++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) S(j,k) += coord[j][i]*coord[k][i]/(error[i]*error[i]);
        
        for(int ijack=0; ijack<njacks; ijack++)
            for(int k=0; k<n_par; k++)
                if(std::isnan(error[i])==0) Sy[ijack](k) += y[ijack][i]*coord[k][i]/(error[i]*error[i]);
    }
    
    for(int ijack=0; ijack<njacks; ijack++)
        jpars[ijack] = S.colPivHouseholderQr().solve(Sy[ijack]);
    
    vvd_t par_array(vd_t(0.0,2),n_par);
    
    vd_t par_ave(0.0,n_par), par2_ave(0.0,n_par), par_err(0.0,n_par);
    
    for(int k=0; k<n_par; k++)
    {
        for(int ijack=0;ijack<njacks;ijack++)
        {
            par_ave[k]+=jpars[ijack](k)/njacks;
            par2_ave[k]+=jpars[ijack](k)*jpars[ijack](k)/njacks;
        }
        par_err[k]=sqrt((double)(njacks-1))*sqrt(fabs(par2_ave[k]-par_ave[k]*par_ave[k]));
        
        par_array[k][0] = par_ave[k];
        par_array[k][1] = par_err[k];
    }
    
    //    if(path!="")
    //    {
    //      ofstream out(path);
    //      out<<"@type xydy"<<endl;
    //      for(int i=1; i<range_max; i++)
    //	out<<i<<" "<<y[0][i]<<" "<<error[i]<<endl;
    //      out<<"&"<<endl;
    //      out<<"@type xy"<<endl;
    //      out<<range_min<<" "<<par_ave[0]-par_err[0]<<endl;
    //      out<<range_min<<" "<<par_ave[0]+par_err[0]<<endl;
    //      out<<range_max<<" "<<par_ave[0]+par_err[0]<<endl;
    //      out<<range_min<<" "<<par_ave[0]-par_err[0]<<endl;
    //      out<<range_max<<" "<<par_ave[0]-par_err[0]<<endl;
    //    }
    
    return par_array;
    
}

