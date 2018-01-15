#include "global.hpp"
#include "aliases.hpp"
#include "jack.hpp"
#include <iostream>
#include <iomanip>

// to string with precision
template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
    ostringstream out;
    out << fixed;
    out << setprecision(n) << a_value;
    return out.str();
}

//create the path-string to the contraction
string path_to_contr(/*const string &string_path,*/ int i_conf, const int mr1, const string &T1, const int mr2, const string &T2)
{
    int r1 = mr1%nr;
    int m1 = (mr1-r1)/nr;
    int r2 = mr2%nr;
    int m2 = (mr2-r2)/nr;
    
    // complete path
    string string_path;
    
    if(strcmp(action.c_str(),"Iwa")==0)
    {
        string_path = path_ensemble_str+action+"_b"+to_string_with_precision(beta,2)+"_L"+to_string(size[1])+"T"+to_string(size[0])+"_k"+to_string_with_precision(kappa,6)+"_mu"+to_string_with_precision(mu_sea,4)+"/";
    }
    if(strcmp(action.c_str(),"Sym")==0)
    {
        string_path = path_ensemble_str+to_string_with_precision(beta,2)+"_"+to_string(size[1])+"_"+to_string_with_precision(mu_sea,4)+"/";
    }

    char path[1024];
    sprintf(path,"%sout/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",string_path.c_str(),i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());
    
    // cout<<path<<endl;
    
    return path;
}


//get the contraction from file
vvd_t get_contraction(const int mr1, const string &T1, const int mr2, const string &T2, const string &ID, const string &reim, const string &parity/*, const int T, const int nconfs, const int njacks , const int* conf_id, const string &string_path*/)
{
    
    int T=size[0];
    
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    vd_t data_V0P5_real(0.0,T);
    vd_t data_V0P5_imag(0.0,T);
    vd_t data_P5P5_real(0.0,T);
    vd_t data_P5P5_imag(0.0,T);
    
    vvd_t jP5P5_real(vd_t(0.0,T),njacks);
    vvd_t jP5P5_imag(vd_t(0.0,T),njacks);
    vvd_t jV0P5_real(vd_t(0.0,T),njacks);
    vvd_t jV0P5_imag(vd_t(0.0,T),njacks);
    
    //int clust_size=nconfs/njacks;
    
    /////////
    
    for(int iconf=0;iconf<nconfs;iconf++)
    {
        int ijack=iconf/clust_size;
        
        ifstream infile;
        
        string path=path_to_contr(/*string_path,*/conf_id[iconf],mr1,T1,mr2,T2);
//        cout<<"opening: "<<path<<endl;
        infile.open(path);
        
        if(!infile.good())
        {
            cerr<<"Unable to open file "<<path_to_contr(/*string_path,*/conf_id[iconf],mr1,T1,mr2,T2)<<endl;
            exit(1);
        }
        
        //DEBUG
        // cout<<"  Reading contraction from "<<path_to_contr(conf_id[iconf],mr1,T1,mr2,T2)<<endl;
        //DEBUG
        
        infile.ignore(256,'5');
        
        for(int t=0; t<T; t++)
        {
            infile>>data_V0P5_real[t];
            infile>>data_V0P5_imag[t];
        }
        
        infile.ignore(256,'5');
        infile.ignore(256,'5');
        
        for(int t=0; t<T; t++)
        {
            infile>>data_P5P5_real[t];
            infile>>data_P5P5_imag[t];
        }
        
        for(int t=0; t<T; t++) jV0P5_real[ijack][t]+=data_V0P5_real[t];
        for(int t=0; t<T; t++) jV0P5_imag[ijack][t]+=data_V0P5_imag[t];
        for(int t=0; t<T; t++) jP5P5_real[ijack][t]+=data_P5P5_real[t];
        for(int t=0; t<T; t++) jP5P5_imag[ijack][t]+=data_P5P5_imag[t];
        
        infile.close();
    }
    
    jV0P5_real=jackknife/*_double*/(jV0P5_real,T/*,nconfs,clust_size*/);
    jV0P5_imag=jackknife(jV0P5_imag,T);
    jP5P5_real=jackknife(jP5P5_real,T);
    jP5P5_imag=jackknife(jP5P5_imag,T);
    
    vvd_t jvec(vd_t(0.0,T),njacks);
    
    if(ID=="P5P5" and reim=="RE") jvec=jP5P5_real;
    if(ID=="P5P5" and reim=="IM") jvec=jP5P5_imag;
    if(ID=="V0P5" and reim=="RE") jvec=jV0P5_real;
    if(ID=="V0P5" and reim=="IM") jvec=jV0P5_imag;
    
    double par;
    
    if(parity=="EVEN") par=1.0;
    if(parity=="ODD") par=-1.0;
    
    vvd_t jvec_sym(vd_t(0.0,T),njacks);
    vvd_t jvec_par(vd_t(0.0,T/2+1),njacks);
    
    for(int ijack=0;ijack<njacks;ijack++)
    {
        for(int t=0;t<T;t++)
            jvec_sym[ijack][(T-t)%T]=jvec[ijack][t];
        for(int t=0;t<T/2+1;t++)
            jvec_par[ijack][t]=(jvec[ijack][t]+par*jvec_sym[ijack][t])/2.0;
    }
    
    string path=ID+"_"+reim+"_mrbw_"+to_string(mr1)+"_mrfw_"+to_string(mr2)+"_"+T1+T2+".xmg";
    ofstream out(path);
    for(int t=0;t<T;t++)
        out<<t<<" "<<jvec_sym[0][t]<<endl;
    
    // if(ID=="P5P5" and reim=="RE" and parity=="EVEN"){
    //   cout<<"**********DEBUG*************"<<endl;
    //   for(int ijack=0;ijack<njacks;ijack++)
    //     for(int t=0;t<T;t++)
    // 	cout<<jvec[ijack][t]<<endl;
    //   cout<<"**********DEBUG*************"<<endl;}
    
    return jvec_par;
    
}
