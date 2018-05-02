#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <omp.h>
#include "operations.hpp"
#include "rotate.hpp"
#include "jack.hpp"
#include "prop.hpp"
#include "print.hpp"
#include "ave_err.hpp"

// function to compute sigma1
vvd_t oper_t::compute_sigma1(jprop_t &jprop_inv, const int imom)
{
    vvd_t sig1(vd_t(0.0,nmr),njacks);
    prop_t p_slash(prop_t::Zero());
    
    int count=0;
    for(int igam=1;igam<5;igam++)
    {
        p_slash+=GAMMA[igam]*p_tilde[imom][igam%4];
        
        if(p_tilde[imom][igam%4]!=0.) count++;
    }
    
    if(UseSigma1==0) // using RI'-MOM definition
    {
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<nmr;mr++)
            {
                sig1[ijack][mr]=(p_slash*jprop_inv[ijack][mr]).trace().imag()/p2_tilde[imom]/(12.*V);
            }
    }
    if(UseSigma1==1) // using "Sigma1" variant
    {
        vvprop_t sig1_tmp(vprop_t(prop_t::Zero(),nmr),njacks);
        
#pragma omp parallel for collapse(2)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<nmr;mr++)
            {
                for(int igam=1;igam<5;igam++)
                    if(p_tilde[imom][igam%4]!=0.)
                    {
                        sig1_tmp[ijack][mr]+=GAMMA[igam]*jprop_inv[ijack][mr]/p_tilde[imom][igam%4];
                    }
                sig1_tmp[ijack][mr]/=(double)count;
                sig1[ijack][mr]=sig1_tmp[ijack][mr].trace().imag()/(12.0*V);
            }
    }

    return sig1;
}

// function to compute sigma2
vvd_t oper_t::compute_sigma2(jprop_t &jprop_inv)
{
    vvd_t sig2(vd_t(0.0,nmr),njacks);
    
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
        {
            sig2[ijack][mr]=(jprop_inv[ijack][mr]*GAMMA[0]).trace().real()/(12.0*V);
        }
    
    return sig2;
}

// function to compute sigma3
vvd_t oper_t::compute_sigma3(jprop_t &jprop_inv)
{
    vvd_t sig3(vd_t(0.0,nmr),njacks);
    
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
        {
            sig3[ijack][mr]=(jprop_inv[ijack][mr]*GAMMA[5]).trace().imag()/(12.0*V);
        }
    
    return sig3;
}

void oper_t::plot_sigmas()
{
    cout<<"  Plotting sigmas"<<endl;
    
    vector<Zq_tup> sigma1_ave_err= {ave_err(sigma1_LO),ave_err(sigma1_PH),ave_err(sigma1_P),ave_err(sigma1_S)};
    vector<Zq_tup> sigma2_ave_err= {ave_err(sigma2_LO),ave_err(sigma2_PH),ave_err(sigma2_P),ave_err(sigma2_S)};
    vector<Zq_tup> sigma3_ave_err= {ave_err(sigma3_LO),ave_err(sigma3_PH),ave_err(sigma3_P),ave_err(sigma3_S)};
    
    vector<vvd_t> sigma1_ave(4), sigma1_err(4);
    vector<vvd_t> sigma2_ave(4), sigma2_err(4);
    vector<vvd_t> sigma3_ave(4), sigma3_err(4);
    
    vector<string> kind={"LO","PH","P","S"};
    vector<ofstream> sigma1_data(4), sigma2_data(4), sigma3_data(4);
    
    for(int i=0;i<4;i++)
    {
        sigma1_data[i].open(path_to_ens+"plots/sigma1_"+kind[i]+".txt");
        sigma2_data[i].open(path_to_ens+"plots/sigma2_"+kind[i]+".txt");
        sigma3_data[i].open(path_to_ens+"plots/sigma2_"+kind[i]+".txt");
        
        sigma1_ave[i]=get<0>(sigma1_ave_err[i]);
        sigma2_ave[i]=get<0>(sigma2_ave_err[i]);
        sigma3_ave[i]=get<0>(sigma3_ave_err[i]);
        
        sigma1_err[i]=get<1>(sigma1_ave_err[i]);
        sigma2_err[i]=get<1>(sigma2_ave_err[i]);
        sigma3_err[i]=get<1>(sigma3_ave_err[i]);

        vector<double> p2t;
        
        if(_linmoms==moms)
        {
            p2t.resize(_linmoms);
            read_vec(p2t,path_print+"p2_tilde.txt");
        }
        else
        {
            p2t.resize(_linmoms);
            read_vec(p2t,path_print+"p2_tilde_eqmoms.txt");
        }
        
        for(int imom=0; imom<_linmoms; imom++)
        {
            sigma1_data[i]<<p2t[imom]<<"\t"<<sigma1_ave[i][imom][0]<<"\t"<<sigma1_err[i][imom][0]<<endl;
            sigma2_data[i]<<p2t[imom]<<"\t"<<sigma2_ave[i][imom][0]<<"\t"<<sigma2_err[i][imom][0]<<endl;
            sigma3_data[i]<<p2t[imom]<<"\t"<<sigma2_ave[i][imom][0]<<"\t"<<sigma3_err[i][imom][0]<<endl;
        }
    }
}

void oper_t::compute_sigmas()
{
    ifstream sigma1_LO_data(path_print+"sigma1_LO");
    ifstream sigma1_PH_data(path_print+"sigma1_PH");
    ifstream sigma1_P_data(path_print+"sigma1_P");
    ifstream sigma1_S_data(path_print+"sigma1_S");

    ifstream sigma2_LO_data(path_print+"sigma2_LO");
    ifstream sigma2_PH_data(path_print+"sigma2_PH");
    ifstream sigma2_P_data(path_print+"sigma2_P");
    ifstream sigma2_S_data(path_print+"sigma2_S");

    ifstream sigma3_LO_data(path_print+"sigma3_LO");
    ifstream sigma3_PH_data(path_print+"sigma3_PH");
    ifstream sigma3_P_data(path_print+"sigma3_P");
    ifstream sigma3_S_data(path_print+"sigma3_S");
    
    if(sigma1_LO_data.good() and sigma1_PH_data.good() and sigma1_P_data.good() and sigma1_S_data.good() and
       sigma2_LO_data.good() and sigma2_PH_data.good() and sigma2_P_data.good() and sigma2_S_data.good() and
       sigma3_LO_data.good() and sigma3_PH_data.good() and sigma3_P_data.good() and sigma3_S_data.good())
    {
        cout<<"Reading Sigmas from files"<<endl<<endl;
        
        read_vec_bin(sigma1_LO,path_print+"sigma1_LO");
        read_vec_bin(sigma1_PH,path_print+"sigma1_PH");
        read_vec_bin(sigma1_P,path_print+"sigma1_P");
        read_vec_bin(sigma1_S,path_print+"sigma1_S");

        read_vec_bin(sigma2_LO,path_print+"sigma2_LO");
        read_vec_bin(sigma2_PH,path_print+"sigma2_PH");
        read_vec_bin(sigma2_P,path_print+"sigma2_P");
        read_vec_bin(sigma2_S,path_print+"sigma2_S");

        read_vec_bin(sigma3_LO,path_print+"sigma3_LO");
        read_vec_bin(sigma3_PH,path_print+"sigma3_PH");
        read_vec_bin(sigma3_P,path_print+"sigma3_P");
        read_vec_bin(sigma3_S,path_print+"sigma3_S");
    }
    else
    {
        cout<<"Computing sigmas -- ";
        
        // array of input files to be read in a given conf
        FILE* input[combo];
        vector<string> v_path = setup_read_qprop(input);
        
        for(int ilinmom=0; ilinmom<_linmoms; ilinmom++)
        {
            cout<<"\r\t linmom = "<<ilinmom+1<<"/"<<_linmoms<<endl;
            
            // definition of jackknifed propagators
            jprop_t jS_LO(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS_PH(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS_S(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            // definition of inverse propagators
            jprop_t jS_inv_LO(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS_inv_PH(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS_inv_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            jprop_t jS_inv_S(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
            
            for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
                for(int ihit=0;ihit<nhits;ihit++)
                {
                    vvvprop_t S=read_qprop_mom(input,v_path,i_in_clust,ihit,ilinmom);
                    S=rotate(S);
                    
                    build_prop(S,jS_LO,jS_PH,jS_P,jS_S);
                }
            
            // jackknife average
            jS_LO = jackknife(jS_LO);
            jS_PH = jackknife(jS_PH);
            jS_P  = jackknife(jS_P);
            jS_S  = jackknife(jS_S);
            
            // invert propagator
            jS_inv_LO = invert_jprop(jS_LO);
            jS_inv_PH = jS_inv_LO*jS_PH*jS_inv_LO;
            jS_inv_P  = jS_inv_LO*jS_P*jS_inv_LO;
            jS_inv_S  = jS_inv_LO*jS_S*jS_inv_LO;
            
            // define the computation of the sigmas
#define COMPUTE_SIGMA(A)						\
sigma1_ ## A[ilinmom]=compute_sigma1(jS_inv_ ## A,ilinmom);	\
sigma2_ ## A[ilinmom]=compute_sigma2(jS_inv_ ## A);	\
sigma3_ ## A[ilinmom]=compute_sigma3(jS_inv_ ## A)
            
            // compute sigma
            COMPUTE_SIGMA(LO);
            COMPUTE_SIGMA(PH);
            COMPUTE_SIGMA(P);
            COMPUTE_SIGMA(S);
            
#undef COMPUTE_SIGMA
        } // close linmoms loop
        
        // define the printing of sigmas
#define PRINT_SIGMA(A)						\
print_vec_bin(sigma1_ ## A, path_print+"sigma1_"#A);	\
print_vec_bin(sigma2_ ## A, path_print+"sigma2_"#A);	\
print_vec_bin(sigma3_ ## A, path_print+"sigma3_"#A)

        // print sigmas
        PRINT_SIGMA(LO);
        PRINT_SIGMA(PH);
        PRINT_SIGMA(P);
        PRINT_SIGMA(S);
        
#undef PRINT_SIGMA
    }
    
    plot_sigmas();
}