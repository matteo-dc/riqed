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
string path_to_contr(const string &suffix, const string &out, const string &string_path, int i_conf, const int mr1, const string &T1, const int mr2, const string &T2)
{
    char path[1024];

    if(suffix.compare("")==0)
    {
        int r1 = mr1%nr;
        int m1 = (mr1-r1)/nr;
        int r2 = mr2%nr;
        int m2 = (mr2-r2)/nr;
        
        sprintf(path,"%s%s/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",string_path.c_str(),out.c_str(),i_conf,m1,r1,T1.c_str(),m2,r2,T2.c_str());
    }
    else if(suffix.compare("sea")==0)
    {
        int r1=mr1;
        int r2=mr2;
        
        sprintf(path,"%s%s/%04d/mes_contr_M%s_R%d_%s_M%s_R%d_%s",string_path.c_str(),out.c_str(),i_conf,suffix.c_str(),r1,T1.c_str(),suffix.c_str(),r2,T2.c_str());
    }
    
    return path;
}


//get the contraction from file
vvd_t get_contraction(const string &suffix, const string &out, const int mr1, const string &T1, const int mr2, const string &T2, const string &ID, const string &reim, const string &parity, const int* conf_id , const string &string_path)
{
    int T=size[0];
    
    vd_t data_real(0.0,T);
    vd_t data_imag(0.0,T);
    
    vvd_t jcorr_real(vd_t(0.0,T),njacks);
    vvd_t jcorr_imag(vd_t(0.0,T),njacks);
    
//    vd_t data_V0P5_real(0.0,T);
//    vd_t data_V0P5_imag(0.0,T);
//    vd_t data_P5P5_real(0.0,T);
//    vd_t data_P5P5_imag(0.0,T);
//    
//    vvd_t jP5P5_real(vd_t(0.0,T),njacks);
//    vvd_t jP5P5_imag(vd_t(0.0,T),njacks);
//    vvd_t jV0P5_real(vd_t(0.0,T),njacks);
//    vvd_t jV0P5_imag(vd_t(0.0,T),njacks);
    
    //int clust_size=nconfs/njacks;
    
    /////////
    
    for(int iconf=0;iconf<nconfs;iconf++)
    {
        printf("%s  %s-%s",ID.c_str(),T1.c_str(),T2.c_str());
        printf("\t conf = %d/%d\n",iconf+1,nconfs);
        
        int ijack=iconf/clust_size;
        
        ifstream infile;
        string path=path_to_contr(suffix,out,string_path,conf_id[iconf],mr1,T1,mr2,T2);
        infile.open(path);
        
        if(!infile.good())
        {
            cerr<<"Unable to open file "<<path_to_contr(suffix,out,string_path,conf_id[iconf],mr1,T1,mr2,T2)<<endl;
            exit(1);
        }
        
        size_t pos;
        string line;
        
        while(infile.good())
        {
            getline(infile,line); // get line from file
            pos=line.find(ID); // search
            if(pos!=string::npos) // string::npos is returned if string is not found
            {
                for(int t=0; t<T; t++)
                {
                    infile>>data_real[t];
                    infile>>data_imag[t];
                }
            }
        }
        
//        if(ID=="P5P5")
//        {
//            infile.ignore(256,'5');
//            
//            
//        }
//        infile.ignore(256,'5');
//        infile.ignore(256,'5');
//        
//        for(int t=0; t<T; t++)
//        {
//            infile>>data_P5P5_real[t];
//            infile>>data_P5P5_imag[t];
//        }
        
        for(int t=0; t<T; t++) jcorr_real[ijack][t]+=data_real[t];
        for(int t=0; t<T; t++) jcorr_imag[ijack][t]+=data_imag[t];

//        for(int t=0; t<T; t++) jV0P5_real[ijack][t]+=data_V0P5_real[t];
//        for(int t=0; t<T; t++) jV0P5_imag[ijack][t]+=data_V0P5_imag[t];
//        for(int t=0; t<T; t++) jP5P5_real[ijack][t]+=data_P5P5_real[t];
//        for(int t=0; t<T; t++) jP5P5_imag[ijack][t]+=data_P5P5_imag[t];
        
        infile.close();
    }
    
    jcorr_real=jackknife(jcorr_real,T);
    jcorr_imag=jackknife(jcorr_imag,T);
    
//    jV0P5_real=jackknife(jV0P5_real,T);
//    jV0P5_imag=jackknife(jV0P5_imag,T);
//    jP5P5_real=jackknife(jP5P5_real,T);
//    jP5P5_imag=jackknife(jP5P5_imag,T);
    
    vvd_t jvec(vd_t(0.0,T),njacks);

    if(reim=="RE") jvec=jcorr_real;
    if(reim=="IM") jvec=jcorr_imag;
    
//    if(ID=="P5P5" and reim=="RE") jvec=jP5P5_real;
//    if(ID=="P5P5" and reim=="IM") jvec=jP5P5_imag;
//    if(ID=="V0P5" and reim=="RE") jvec=jV0P5_real;
//    if(ID=="V0P5" and reim=="IM") jvec=jV0P5_imag;
    
    double par;
    
    if(parity=="EVEN") par=+1.0;
    if(parity=="ODD")  par=-1.0;
    if(parity=="UNK")  par=+0.0;
    
    vvd_t jvec_sym(vd_t(0.0,T),njacks);
    vvd_t jvec_par(vd_t(0.0,T/2+1),njacks);
    
    for(int ijack=0;ijack<njacks;ijack++)
    {
        for(int t=0;t<T;t++)
            jvec_sym[ijack][(T-t)%T]=jvec[ijack][t];
        for(int t=0;t<T/2+1;t++)
            jvec_par[ijack][t]=(jvec[ijack][t]+par*jvec_sym[ijack][t])/(1+abs(par));
    }
    
    return jvec_par;
    
}
