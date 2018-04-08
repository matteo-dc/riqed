#include "aliases.hpp"
#include "global.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include "operations.hpp"
#include "read.hpp"
#include "ave_err.hpp"
#include "contractions.hpp"


// read input mom file
void oper_t::read_mom_list(const string &path)
{
    ifstream input(path);
    if(!input.good())
    {
        cerr<<"Error opening "<<path<<endl;
        exit(1);
    }
    
    while(!input.eof())
    {
        coords_t mom;
        p_t p_array, p_tilde_array;
        double p_sqr=0.0, p_tilde_sqr=0.0;
        double p_4=0.0, p_tilde_4=0.0;
        p_t shift;
        if(BC.compare("Periodic")==0) shift={0.0,0.0,0.0,0.0};
        if(BC.compare("Antiperiodic")==0) shift={0.5,0.0,0.0,0.0};
        
        for(int mu=0;mu<4;mu++)
        {
            input>>mom[mu];
            
            p_array[mu]=2*M_PI*(mom[mu]+shift[mu])/size[mu];
            p_sqr+=p_array[mu]*p_array[mu];
            p_4+=p_array[mu]*p_array[mu]*p_array[mu]*p_array[mu];

            p_tilde_array[mu]=sin(p_array[mu]);
            p_tilde_sqr+=p_tilde_array[mu]*p_tilde_array[mu];
            p_tilde_4+=p_tilde_array[mu]*p_tilde_array[mu]*p_tilde_array[mu]*p_tilde_array[mu];
        }
        if(input.good())
        {
            mom_list.push_back(mom);
            p.push_back(p_array);
            p_tilde.push_back(p_tilde_array);
            p2.push_back(p_sqr);
            p2_tilde.push_back(p_tilde_sqr);
            p4.push_back(p_4);
            p4_tilde.push_back(p_tilde_4);
            
            const double D4=p_tilde_4/(p_tilde_sqr*p_tilde_sqr);
            const bool filt=(D4<thresh);
            filt_moms.push_back(filt);

        }
    }
}

// read plaquette
double read_plaquette(const string &path)
{
    double plaquette=0.0;
    
    ifstream input_plaquette;
    input_plaquette.open(path+"plaquette.txt",ios::in);
    if(not input_plaquette.good())
    {
        cout<<"Unable to open \"plaquette.txt\"."<<endl<<endl;
        exit(1);
    }
    
    cout<<"Reading from \"plaquette.txt\"."<<endl<<endl;
    
    for(int iconf=0;iconf<nconfs;iconf++)
    {
        double temp=0.0;
        input_plaquette>>temp;
        plaquette += temp/nconfs;
    }
    input_plaquette.close();
    
    return plaquette;
}

// read effective mass
vvvd_t oper_t::read_eff_mass(const string name)
{
    vvvd_t eff_mass_tmp(vvd_t(vd_t(0.0,nmr),nmr),njacks);

    FILE* input_effmass;
    input_effmass = fopen(name.c_str(),"rb");
    
    if(input_effmass == NULL)
    {
        cout<<"Computing effective masses"<<endl<<endl;
        compute_eff_mass();
        input_effmass = fopen(name.c_str(),"rb");
    }
    
    cout<<"Reading eff_mass"<<endl<<endl;
    
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            {
                double temp;
                
                int rd=fread(&temp,sizeof(double),1,input_effmass);
                if(rd!=1)
                {
                    cerr<<"Unable to read from \""<<name<<"\" mr_fw: "<<mr_fw<<", mr_bw: "<<mr_bw<<", ijack: "<<ijack<<endl;
                    exit(1);
                }
                eff_mass_tmp[ijack][mr_fw][mr_bw]=temp; //store
            }
    
    vvd_t eff_mass_ave=get<0>(ave_err(eff_mass_tmp));
    vvd_t eff_mass_err=get<1>(ave_err(eff_mass_tmp));
    
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            {
                printf("mr1: %d \t mr2: %d \t %lg +- %lg\n",mr_fw,mr_bw,eff_mass_ave[mr_fw][mr_bw],eff_mass_err[mr_fw][mr_bw]);
            }
    printf("\n");
    
    return eff_mass_tmp;
}

// read effective mass time dependent
vvvvd_t oper_t::read_eff_mass_time(const string name)
{
    int T=size[0];
    vvvvd_t eff_mass_time_tmp(vvvd_t(vvd_t(vd_t(0.0,T/2),njacks),nmr),nmr);
    
    FILE* input_effmass_time;
    input_effmass_time = fopen(name.c_str(),"rb");
    
    if(input_effmass_time == NULL)
    {
        cout<<"Cannot read eff_mass_array_time."<<endl<<endl;
        exit(1);
    }
    
    cout<<"Reading eff_mass_array_time"<<endl<<endl;
    
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int t=0;t<T/2;t++)
                {
                    double temp;
                    
                    int rd=fread(&temp,sizeof(double),1,input_effmass_time);
                    if(rd!=1)
                    {
                        cerr<<"Unable to read from \""<<name<<"\" mr_fw: "<<mr_fw<<", mr_bw: "<<mr_bw<<", ijack: "<<ijack<<", t: "<<t<<endl;
                        exit(1);
                    }
                    eff_mass_time_tmp[mr_fw][mr_bw][ijack][t]=temp; //store
                }
    
//    vvd_t eff_mass_ave=get<0>(ave_err(eff_mass_tmp));
//    vvd_t eff_mass_err=get<1>(ave_err(eff_mass_tmp));
//    
//    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
//        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
//        {
//            printf("mr1: %d \t mr2: %d \t %lg +- %lg\n",mr_fw,mr_bw,eff_mass_ave[mr_fw][mr_bw],eff_mass_err[mr_fw][mr_bw]);
//        }
//    printf("\n");
    
    return eff_mass_time_tmp;
}


// read effective sea mass
vvvd_t oper_t::read_eff_mass_sea(const string name)
{
    vvvd_t eff_mass_sea_tmp(vvd_t(vd_t(0.0,nr),nr),njacks);
    
    FILE* input_effmass_sea;
    input_effmass_sea = fopen(name.c_str(),"rb");
    
    if(input_effmass_sea == NULL)
    {
        cout<<"Computing effective sea masses"<<endl<<endl;
        compute_eff_mass_sea();
        input_effmass_sea = fopen(name.c_str(),"rb");
    }
    
    cout<<"Reading eff_mass sea"<<endl<<endl;
    
    for(int ijack=0;ijack<njacks;ijack++)
        for(int r1=0;r1<nr;r1++)
            for(int r2=0;r2<nr;r2++)
            {
                double temp;
                
                int rd=fread(&temp,sizeof(double),1,input_effmass_sea);
                if(rd!=1)
                {
                    cerr<<"Unable to read from \""<<name<<"\" r1: "<<r1<<", r2: "<<r2<<", ijack: "<<ijack<<endl;
                    exit(1);
                }
                eff_mass_sea_tmp[ijack][r1][r2]=temp; //store
            }
    
    vvd_t eff_mass_sea_ave=get<0>(ave_err(eff_mass_sea_tmp));
    vvd_t eff_mass_sea_err=get<1>(ave_err(eff_mass_sea_tmp));
    
    for(int r1=0;r1<nr;r1++)
        for(int r2=0;r2<nr;r2++)
        {
            printf("r1: %d \t r2: %d \t %lg +- %lg\n",r1,r2,eff_mass_sea_ave[r1][r2],eff_mass_sea_err[r1][r2]);
        }
    printf("\n");

    return eff_mass_sea_tmp;
}


// read deltamu and deltamcr
vvvvd_t oper_t::read_deltam(const string path, const string name)
{
    vvvvd_t deltam(vvvd_t(vvd_t(vd_t(0.0,nr),nm),nm),njacks);
    
    FILE* input_deltam;
    input_deltam = fopen((path+name).c_str(),"rb");
    
     if(input_deltam == NULL)
     {
         cout<<"Computing "<<name<<endl<<endl;
         compute_deltam();
         input_deltam = fopen((path+name).c_str(),"rb");
     }
    
    cout<<"Reading "<<name<<endl<<endl;
    
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int r=0;r<nr;r++)
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    double temp;
                    
                    int rd=fread(&temp,sizeof(double),1,input_deltam);
                    if(rd!=1)
                    {
                        cerr<<"Unable to read from \""<<name<<"\" m_fw: "<<m_fw<<", m_bw: "<<m_bw<<", r: "<<r<<", ijack: "<<ijack<<endl;
                        exit(1);
                    }
                    
                    deltam[ijack][m_fw][m_bw][r]=temp; //store
                }
    
    vvvd_t deltam_ave=get<0>(ave_err(deltam));
    vvvd_t deltam_err=get<1>(ave_err(deltam));
   
    for(int m_fw=0;m_fw<nm;m_fw++)
        for(int m_bw=0;m_bw<nm;m_bw++)
            for(int r=0;r<nr;r++)
                printf("m1: %d \t m2: %d \t r: %d \t %lg +- %lg\n",m_fw,m_bw,r,deltam_ave[m_fw][m_bw][r],deltam_err[m_fw][m_bw][r]);
    printf("\n");

    return deltam;
}

// returns the linearized spin color index
size_t isc(size_t is,size_t ic)
{return ic+3*is;}

// creates the path-string to the configuration
string path_to_conf(const string &string_path, const string &out, int i_conf,const string &name)
{
    char path[1024];
    sprintf(path,"%s%s/%04d/fft_%s",string_path.c_str(),out.c_str(),i_conf,name.c_str());
    
    return path;
}

// opens all the files and return a vector with all the string paths
vector<string> oper_t::setup_read_qprop(FILE* input[])
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    // create propagator name
    vector<string> Mass(nm);
    for (int im=0; im<nm; im++)
        Mass[im]="M"+to_string(im)+"_";
    vector<string> R(nr);
    for (int ir=0; ir<nr; ir++)
        R[ir]="R"+to_string(ir)+"_";
    vector<string> Types;
    if(ntypes==5) Types={"0","F","FF","T","P"};
    if(ntypes==6) Types={"0","F","FF","T","P","S"};
    
    vector<string> v_path;
    
    // open files
    //#pragma omp parallel for collapse(5)
    for(int iconf=0;iconf<nconfs;iconf++)
        for(int ihit=0;ihit<nhits;ihit++)
            for(int t=0;t<ntypes;t++)
                for(int m=0;m<nm;m++)
                    for(int r=0;r<nr;r++)
                    {
                        string hit_suffix = "";
                        if(nhits>1) hit_suffix = "_hit_" + to_string(ihit);
                        
                        int icombo=r + nr*m + nr*nm*t + nr*nm*ntypes*ihit + nr*nm*ntypes*nhits*iconf;
                        
                        string path = path_to_conf(path_to_ens,out_hadr,conf_id[iconf],"S_"+Mass[m]+R[r]+Types[t]+hit_suffix);
                        v_path.push_back(path);
                        
                        input[icombo] = fopen(path.c_str(),"rb");
                        
                        if(input[icombo]==NULL)
                        {
                            fprintf(stderr,"Unable to open file %s - combo %d / %d \n", path.c_str(), icombo, combo);
                            exit(1);
                        }
                    }
    
    printf("Opened all the files in %s\n",path_to_ens.c_str());
    
    return v_path;
}

vector<string> oper_t::setup_read_lprop(FILE* input[])
{
    // array of the configurations
    int conf_id[nconfs];
    for(int iconf=0;iconf<nconfs;iconf++)
        conf_id[iconf]=conf_init+iconf*conf_step;
    
    // create propagator name
    vector<string> Types={"0","F"};
    vector<string> v_path;
    
    const int ntypes_lep=2;
    
    // open files
    //#pragma omp parallel for collapse(5)
    for(int iconf=0;iconf<nconfs;iconf++)
        for(int ihit=0;ihit<nhits;ihit++)
            for(int t=0;t<ntypes_lep;t++)
            {
                string hit_suffix = "";
                if(nhits>1) hit_suffix = "_hit_" + to_string(ihit);
                
                int icombo=t + ntypes_lep*ihit + ntypes_lep*nhits*iconf;
                
                string path = path_to_conf(path_to_ens,out_lep,conf_id[iconf],"L_"+Types[t]+hit_suffix);
                v_path.push_back(path);
                
                input[icombo] = fopen(path.c_str(),"rb");
                
                if(input[icombo]==NULL)
                {
                    fprintf(stderr,"Unable to open file %s - combolep %d / %d \n", path.c_str(), icombo, combo);
                    exit(1);
                }
            }
    
    printf("Opened all the files in %s\n",path_to_ens.c_str());
    
    return v_path;
}


//read a propagator file
prop_t read_prop(FILE* input, const string &path, const int imom, const dcompl fact)
{
    prop_t out(prop_t::Zero());
    
    // start to read from the momentum imom
    long offset = imom*sizeof(dcompl)*4*4*3*3;
    fseek(input,offset,SEEK_SET);
    
    for(int id_so=0;id_so<4;id_so++)
        for(int ic_so=0;ic_so<3;ic_so++)
            for(int id_si=0;id_si<4;id_si++)
                for(int ic_si=0;ic_si<3;ic_si++)
                {
                    double temp[]={0.0,0.0};
                  
                    if(input==NULL)
                    {
                        cerr<<"Bad before reading"<<endl;
                        exit(1);
                    }
                    
                    int rc=fread(&temp,sizeof(double)*2,1,input);
                    if(rc!=1)
                    {
                        cerr<<"Unable to read from "<<path<<" id_so: "<<id_so<<", ic_so: "<<ic_so<<", id_si: "<<id_si<<", ic_si:"<<ic_si<<endl;
                        exit(1);
                    }
                    out(isc(id_si,ic_si),isc(id_so,ic_so))=fact*dcompl(temp[0],temp[1]); //store
                }
    
    return out;
}

//read all the quark propagators at a given momentum
vvvprop_t read_qprop_mom(FILE* input[],const vector<string> v_path,const int i_in_clust,const int ihit,const int imom)
{
    vvvprop_t S(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);

#pragma omp parallel for
    for(int ilin=0;ilin<nm*nr*ntypes*njacks;ilin++)
    {
        int k=ilin;
        int r = k % nr;
        k/=nr;
        int m = k % nm;
        k/=nm;
        int t = k % ntypes;
        k/=ntypes;
        int ijack = k % njacks;

        int mr = r + nr*m;
        
        int iconf=clust_size*ijack+i_in_clust;
        int icombo = r + nr*m + nr*nm*t + nr*nm*ntypes*ihit + nr*nm*ntypes*nhits*iconf;
        
        //create all the propagators in a given conf and a given mom
        S[ijack][t][mr] = read_prop(input[icombo],v_path[icombo],imom,coeff_to_read(t,r));
    }
    
    return S;
}

// read all the lepton propagators at a given momentum
vvprop_t read_lprop_mom(FILE* input[],const vector<string> v_path,const int i_in_clust,const int ihit,const int imom)
{
    vvprop_t L(vprop_t(prop_t::Zero(),ntypes_lep),njacks);
    
#pragma omp parallel for
    for(int ilin=0;ilin<ntypes_lep*njacks;ilin++)
    {
        int k=ilin;
        int t = k % ntypes_lep;
        k/=ntypes_lep;
        int ijack = k % njacks;
        
        int iconf=clust_size*ijack+i_in_clust;
        int icombo = t + ntypes_lep*ihit + ntypes_lep*nhits*iconf;
        
        //create all the propagators in a given conf and a given mom
        L[ijack][t] = read_prop(input[icombo],v_path[icombo],imom,1.0);
    }
    
    return L;
}



//read file
void read_internal(double &t,ifstream& infile)
{
    infile>>t;
}
void read_internal(VectorXd &V, ifstream& infile)
{
    for(int i=0; i<V.size();i++) read_internal(V(i),infile);
}

//read file binary
void read_internal_bin(VectorXd &V, ifstream& infile)
{
    for(int i=0; i<V.size();i++) read_internal(V(i),infile);
}

