#include "aliases.hpp"
#include "global.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <iomanip>
#include "effective_mass.hpp"
#include "deltam_cr.hpp"
#include <omp.h>


// read input mom file
void read_mom_list(const string &path)
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
        if(BC_str.compare("Periodic")==0) shift={0.0,0.0,0.0,0.0};
        if(BC_str.compare("Antiperiodic")==0) shift={0.5,0.0,0.0,0.0};
        
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
        }
    }
}

// read plaquette
double read_plaquette()
{
    double plaquette=0.0;
    
    ifstream input_plaquette;
    input_plaquette.open("plaquette.txt",ios::in);
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
vvd_t read_eff_mass(const string name)
{
    vvvd_t eff_mass_array(vvd_t(vd_t(0.0,2),nmr),nmr);
    
    ifstream input_effmass;
    input_effmass.open(name,ios::binary);
    
    if(not input_effmass.good())
    {
        cout<<"Computing effective masses"<<endl<<endl;
        compute_eff_mass();
        input_effmass.open(name,ios::binary);
    }
    
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            for(int i=0;i<2;i++)
            {
                double temp;
                input_effmass.read((char*)&temp,sizeof(double));
                if(not input_effmass.good())
                {
                    cerr<<"Unable to read from \"eff_mass_array\" mr_fw: "<<mr_fw<<", mr_bw: "<<mr_bw<<", i: "<<i<<endl;
                    exit(1);
                }
                eff_mass_array[mr_fw][mr_bw][i]=temp; //store  [i=ave/err]
            }
    
    vvd_t eff_mass(vd_t(0.0,nmr),nmr);
//    eff_mass.resize(nmr);
//    for(auto &i : eff_mass) i.resize(nmr);
    for(int mr_fw=0;mr_fw<nmr;mr_fw++)
        for(int mr_bw=0;mr_bw<nmr;mr_bw++)
            eff_mass[mr_fw][mr_bw] = eff_mass_array[mr_fw][mr_bw][0];
    
    return eff_mass;
}

// read deltam_cr
vvvd_t read_deltam_cr(const string name)
{
    vvvd_t deltam_cr(vvd_t(vd_t(0.0,nm),nm),njacks);
    
    ifstream input_deltam;
    input_deltam.open(name,ios::binary);

     if(not input_deltam.good())
     {
         cout<<"Computing deltam_cr"<<endl<<endl;
         compute_deltam_cr();
         input_deltam.open(name,ios::binary);
     }
    
    cout<<"Reading deltam_cr"<<endl<<endl;
    
    for(int ijack=0;ijack<njacks;ijack++)
        for(int m_fw=0;m_fw<nm;m_fw++)
            for(int m_bw=0;m_bw<nm;m_bw++)
            {
                double temp;
                input_deltam.read((char*)&temp,sizeof(double));
                if(not input_deltam.good())
                {
                    cerr<<"Unable to read from \"deltam_cr_array\" mr_fw: "<<m_fw<<", mr_bw: "<<m_bw<<", ijack: "<<ijack<<endl;
                    exit(1);
                }
                deltam_cr[ijack][m_fw][m_bw]=temp; //store
            }
    return deltam_cr;
}

// returns the linearized spin color index
size_t isc(size_t is,size_t ic)
{return ic+3*is;}

// to string with precision
template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
    ostringstream out;
    out << fixed;
    out << setprecision(n) << a_value;
    return out.str();
}

// creates the path-string to the configuration
string path_to_conf(const string &string_path, int i_conf,const string &name)
{
    char path[1024];
    sprintf(path,"%sout/%04d/fft_%s",string_path.c_str(),i_conf,name.c_str());
    
    return path;
}

// opens all the files and return a vector with all the string paths
vector<string> setup_read_prop(ifstream *input)
{
    // complete path to conf
    string string_path;
    if(strcmp(action.c_str(),"Iwa")==0)
    {
        string_path = path_ensemble_str+action+"_b"+to_string_with_precision(beta,2)+"_L"+to_string(size[1])+"T"+to_string(size[0])+"_k"+to_string_with_precision(kappa,6)+"_mu"+to_string_with_precision(mu_sea,4)+"/";
    }
    if(strcmp(action.c_str(),"Sym")==0)
    {
        string_path = path_ensemble_str+to_string_with_precision(beta,2)+"_"+to_string(size[1])+"_"+to_string_with_precision(mu_sea,4)+"/";
    }
    
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
                        
                        string path = path_to_conf(string_path,conf_id[iconf],"S_"+Mass[m]+R[r]+Types[t]+hit_suffix);
                        v_path.push_back(path);
                        
                        input[icombo].open(path,ios::binary);
                        
                        if(!input[icombo].good())
                        {
                            fprintf(stderr,"Unable to open file %s - combo %d / %d \n", path.c_str(), icombo, combo);
                            exit(1);
                        }
                    }
    
    printf("Opened all the files in %s\n",string_path.c_str());
    
    return v_path;
}


//read a propagator file
prop_t read_prop(ifstream &input, const string &path, const int imom)
{
    prop_t out(prop_t::Zero());
    
    // start to read from the momentum imom
    long offset = imom*sizeof(dcompl)*4*4*3*3;
    input.seekg(offset,input.beg);
    
    for(int id_so=0;id_so<4;id_so++)
        for(int ic_so=0;ic_so<3;ic_so++)
            for(int id_si=0;id_si<4;id_si++)
                for(int ic_si=0;ic_si<3;ic_si++)
                {
                    double temp[2];
                    if(not input.good())
                    {
                        cerr<<"Bad before reading"<<endl;
                        exit(1);
                    }
                    input.read((char*)&temp,sizeof(double)*2);
                    if(not input.good())
                    {
                        cerr<<"Unable to read from "<<path<<" id_so: "<<id_so<<", ic_so: "<<ic_so<<", id_si: "<<id_si<<", ic_si:"<<ic_si<<endl;
                        exit(1);
                    }
                    out(isc(id_si,ic_si),isc(id_so,ic_so))=dcompl(temp[0],temp[1]); //store
                }
    
    return out;
}

//read all the propagators at a given momentum
vvvprop_t read_prop_mom(ifstream *input,const vector<string> v_path,const int i_in_clust,const int ihit,const int imom)
{
    vvvprop_t S(vvprop_t(vprop_t(prop_t::Zero(),nmr),ntypes),njacks);
    
#pragma omp parallel for collapse(4)
    for(int t=0;t<ntypes;t++)
        for(int m=0;m<nm;m++)
            for(int r=0;r<nr;r++)
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    if(omp_get_thread_num()==0)
                    {
                        cout<<" Thread "<<omp_get_thread_num()<<"/"<<omp_get_num_threads()<<" --";
                        cout<<" ijack "<<ijack;
                        cout<<" m "<<m;
                        cout<<" r "<<r;
                        cout<<" t "<<t<<endl;
                    }
                    
                    int iconf=clust_size*ijack+i_in_clust;
                    int icombo=r + nr*m + nr*nm*t + nr*nm*ntypes*ihit + nr*nm*ntypes*nhits*iconf;
                    
                    int mr = r + nr*m; // M0R0,M0R1,M1R0,M1R1,M2R0,M2R1,M3R0,M3R1
                    
                    //printf("  Reading propagator from %s\n",path.c_str());
                    
                    //create all the propagators in a given conf and a given mom
                    S[ijack][t][mr] = read_prop(input[icombo],v_path[icombo],imom);
                    
                    if(t==4) S[ijack][t][mr]*=dcompl(0.0,1.0);      // i*(pseudoscalar insertion)
                    //if(t==5) S[ijack][t][mr]*=dcompl(1.0,0.0);    // (minus sign?)
                }
    return S;
}

//read file
void read_internal(double &t,ifstream& infile)
{
    //infile.read((char*) &t,sizeof(double));
    infile>>t;
}
//template <class T>
void read_internal(VectorXd &V, ifstream& infile)
{
    for(int i=0; i<V.size();i++) read_internal(V(i),infile);
}



