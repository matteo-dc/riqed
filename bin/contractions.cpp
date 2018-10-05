#include "global.hpp"
#include "aliases.hpp"
#include "jack.hpp"
#include <iostream>
#include <iomanip>
#include "contractions.hpp"
#include "tools.hpp"


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
string path_to_contr(const string &suffix, const string &out, const string &string_path, int i_conf, const int m_fw, const int m_bw, const int r_fw, const int r_bw, const string &Tfw, const string &Tbw)
{
    char path[1024];

    if(suffix.compare("")==0)
    {
        sprintf(path,"%s%s/%04d/mes_contr_M%d_R%d_%s_M%d_R%d_%s",string_path.c_str(),out.c_str(),i_conf,m_bw,r_bw,Tbw.c_str(),m_fw,r_fw,Tfw.c_str());
    }
    else if(suffix.compare("sea")==0)
    {
        sprintf(path,"%s%s/%04d/mes_contr_M%s_R%d_%s_M%s_R%d_%s",string_path.c_str(),out.c_str(),i_conf,suffix.c_str(),r_bw,Tbw.c_str(),suffix.c_str(),r_fw,Tfw.c_str());
    }
    
    return path;
}

//compute the coefficient
dcompl coeff_to_read(const size_t ikind,const size_t r)
{
    const size_t nmax=ntypes;
    if(ikind>=nmax)
    {printf("cannot ask for coeff of kind %zu, maximum is %zu",ikind,nmax); exit(1);}
    
    if(ikind==4) //P
        return dcompl(0,tau3[r]);
    else
        if(ikind==5) //S
            return -1.0;
        else //others
            return 1.0;
}


//get the contraction from file
vvd_t get_contraction(const string &suffix, const string &out, const int m_fw, const int m_bw, const int rfw, const int rbw,size_t kfw, size_t kbw, const string &ID, const size_t ext_reim, const int &tpar, const int* conf_id , const string &string_path)
{
    int T=size[0];
    
    vector<string> kind_tag;
    if(ntypes==6) kind_tag={"0","F","FF","T","P","S"};
    if(ntypes==3) kind_tag={"0","F","QED"};
    
    const string Tbw=kind_tag[kbw];
    const string Tfw=kind_tag[kfw];
        
    //Compute the coefficient
    dcompl c_coeff=conj(coeff_to_read(kbw,rbw))*coeff_to_read(kfw,rfw);
    
    //Include -i if asking the imaginary part
    if(ext_reim==1) c_coeff*=dcompl(0.0,-1.0);
    
    if(real(c_coeff)!=0 and imag(c_coeff)!=0)
    {cout<<"Don't know what to do"; exit(1);}
    
    //Find whether to take real or imaginary, and the coefficient
    size_t reim;
    double coeff;
    if(real(c_coeff)!=0)
    {
        reim=0;
        coeff=real(c_coeff);
    }
    else
    {
        reim=1;
        coeff=-imag(c_coeff);
    }
    
    vd_t data(0.0,T);
    vvd_t jcorr(vd_t(0.0,T),njacks);
    
    for(int iconf=0;iconf<nconfs;iconf++)
    {
        if(iconf==0) printf("%s  %s-%s\n",ID.c_str(),Tbw.c_str(),Tfw.c_str());
        //printf("\t conf = %d/%d\n",iconf+1,nconfs);
        
        int ijack=iconf/clust_size;
        
        ifstream infile;
        string path=path_to_contr(suffix,out,string_path,conf_id[iconf],m_fw,m_bw,rfw,rbw,Tfw,Tbw);
        infile.open(path);
        
        if(!infile.good())
        {
            cerr<<"Unable to open file "<<path_to_contr(suffix,out,string_path,conf_id[iconf],m_fw,m_bw,rfw,rbw,Tfw,Tbw)<<endl;
            exit(1);
        }
        
        size_t pos;
        string line;
        
        double tmp;
        
        while(infile.good())
        {
            getline(infile,line); // get line from file
            pos=line.find(ID); // search
            if(pos!=string::npos) // string::npos is returned if string is not found
            {
                for(int t=0; t<T; t++)
                {
                    if(reim==0)
                    {
                        infile>>data[t];
                        infile>>tmp;
                    }
                    else
                    {
                        infile>>tmp;
                        infile>>data[t];
                    }
                }
            }
        }
        
        for(int t=0; t<T; t++) jcorr[ijack][t]+=coeff*data[t];
       
        infile.close();
    }
    
    jcorr=jackknife(jcorr);
    
    vvd_t jcorr_sym(vd_t(0.0,T),njacks);
    vvd_t jcorr_par(vd_t(0.0,T/2+1),njacks);
    
    for(int ijack=0;ijack<njacks;ijack++)
        jcorr_par[ijack]=symmetrize(jcorr[ijack],tpar);
    
    if(abs(tpar))  return jcorr_par;  //dimension T/2+1
    else           return jcorr;      //dimension T

    return jcorr_par; //dimension T/2+1
}
