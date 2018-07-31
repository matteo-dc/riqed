#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <omp.h>
#include "operations.hpp"
#include "rotate.hpp"
#include "jack.hpp"
#include "print.hpp"
#include "ave_err.hpp"
#include "prop.hpp"

#define EXTERN_SIGMA
 #include "sigmas.hpp"

namespace sigma
{
    void set_ins()
    {
        if(ntypes==6)
        {
            ins_list={LO,PH,P,S};
            ins_tag={"LO","PH","P","S"};
        }
        if(ntypes==3)
        {
            ins_list={LO,QED};
            ins_tag={"LO","QED"};
        }
        
        proj_list={SIGMA1,SIGMA2,SIGMA3};
        proj_tag={"1","2","3"};
        
        nproj=proj_list.size();
        nins=ins_list.size();
        nsigma=nproj*nins;
    }
}


// function to compute sigma1, sigma2 or sigma3
vvvd_t oper_t::compute_sigma(vvvprop_t &jprop_inv, const int iproj, const int imom)
{
    using namespace sigma;
    
    vvvd_t sig(vvd_t(vd_t(0.0,nmr),njacks),nins);
    
    switch(iproj)
    {
        case SIGMA1:
        {
            prop_t p_slash(prop_t::Zero());
            int count=0;
            for(int igam=1;igam<5;igam++)
            {
                p_slash+=GAMMA[igam]*p_tilde[imom][igam%4];
                if(p_tilde[imom][igam%4]!=0.) count++;
            }
            
            if(UseSigma1==0) // using RI'-MOM definition
            {
#pragma omp parallel for collapse(3)
                for(int i=0; i<nins; i++)
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int mr=0;mr<nmr;mr++)
                            sig[i][ijack][mr]=(p_slash*jprop_inv[i][ijack][mr]).trace().imag()/p2_tilde[imom]/(12.*V);
            }
            if(UseSigma1==1) // using "Sigma1" variant
            {
                prop_t sig_tmp(prop_t::Zero());
                
#pragma omp parallel for collapse(3)
                for(int i=0; i<nins; i++)
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int mr=0;mr<nmr;mr++)
                        {
                            prop_t sig_tmp(prop_t::Zero());
                            for(int igam=1;igam<5;igam++)
                                if(p_tilde[imom][igam%4]!=0.)
                                {
                                    sig_tmp+=GAMMA[igam]*jprop_inv[i][ijack][mr]/p_tilde[imom][igam%4];
                                }
                            sig_tmp/=(double)count;
                            sig[i][ijack][mr]=sig_tmp.trace().imag()/(12.0*V);
                        }
            }
        }
            break;
        case SIGMA2:
#pragma omp parallel for collapse(3)
            for(int i=0; i<nins; i++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr=0;mr<nmr;mr++)
                    {
                        sig[i][ijack][mr]=(jprop_inv[i][ijack][mr]*GAMMA[0]).trace().real()/(12.0*V);
                    }
            break;
        case SIGMA3:
#pragma omp parallel for collapse(3)
            for(int i=0; i<nins; i++)
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr=0;mr<nmr;mr++)
                    {
                        sig[i][ijack][mr]=(jprop_inv[i][ijack][mr]*GAMMA[5]).trace().imag()/(12.0*V);
                    }
    }
    return sig;
}

void oper_t::plot_sigmas()
{
    using namespace sigma;
    
    cout<<" Plotting sigmas"<<endl<<endl;
    
    // opening files
    vector<ofstream> sigma_data(nsigma);
    for(int j=0; j<nsigma; j++)
    {
        int iproj = j%nproj;
        int ins = (j-iproj)/nproj;
        
        sigma_data[j].open(path_to_ens+"plots/sigma"+proj_tag[iproj]+"_"+ins_tag[ins]+".txt");
    }
    
    // reading p2
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
    
    // plotting averaged sigma
    sigma_tup sigma_ave_err=ave_err(sigma);
    vvvvd_t sigma_ave=get<0>(sigma_ave_err);
    vvvvd_t sigma_err=get<1>(sigma_ave_err);

    for(int j=0; j<nsigma; j++)
    {
        int iproj = j%nproj;
        int ins = (j-iproj)/nproj;
        
        for(int imom=0;imom<_linmoms;imom++)
            sigma_data[j]<<p2t[imom]<<"\t"
                         <<sigma_ave[imom][iproj][ins][0]<<"\t"
                         <<sigma_err[imom][iproj][ins][0]<<endl;
    }
}

void oper_t::compute_sigmas()
{
    using namespace sigma;
    
    ifstream sigma_data;
    sigma_data.open(path_print+"sigmas");
    
    if(sigma_data.good())
    {
        cout<<"Reading Sigmas from files: \""<<path_print<<"\"sigmas"<<endl;
        read_vec_bin(sigma,path_print+"sigmas");
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
            jprop_t jS(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            
            // definition of inverse propagators
            jprop_t jS_inv(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            
            for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
                for(int ihit=0;ihit<nhits;ihit++)
                {
                    vvvprop_t S=read_qprop_mom(input,v_path,i_in_clust,ihit,ilinmom);
                    S=rotate(S);
                    
                    build_prop(S,jS);
                }
            
            // jackknife average
            for(auto &prop : jS)
                prop = jackknife(prop);
            
            // invert propagator
            jS_inv[jprop::LO] = invert_jprop(jS[jprop::LO]);
            for(int i=1;i<jprop::nins;i++)
                jS_inv[i] = - jS_inv[jprop::LO]*jS[i]*jS_inv[jprop::LO];
            
            for(int iproj=0;iproj<nproj;iproj++)
                sigma[ilinmom][iproj]=compute_sigma(jS_inv,iproj,ilinmom);
        } // close linmoms loop
        
        // print sigmas
        print_vec_bin(sigma,path_print+"sigmas");
    }
    
    plot_sigmas();
}