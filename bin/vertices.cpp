#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include "contractions.hpp"
#include <omp.h>
#include "prop.hpp"
#include "rotate.hpp"
#include "jack.hpp"
#include "print.hpp"
#include "prop.hpp"
#include "operations.hpp"
#include <chrono>

#define EXTERN_VERT
 #include "vertices.hpp"

using namespace std::chrono;

namespace gbil
{
    void set_ins()
    {
        if(ntypes==6)
        {
            ins_list={LO,PH,Pfw,Pbw,Sfw,Sbw};
            ins_tag={"LO","EM","Pfw","Pbw","Sfw","Sbw"};
        }
        if(ntypes==3)
        {
            ins_list={LO,QED};
            ins_tag={"LO","QED"};
        }
        
        nins=ins_list.size();
        nGamma=16;
    }
}

//calculate the vertex function in a given configuration for the given equal momenta
prop_t make_vertex(const prop_t &prop1, const prop_t &prop2, const int mu)
{
    return prop1*GAMMA[mu]*GAMMA[5]*prop2.adjoint()*GAMMA[5];
}

// compute LO and EM vertices
void build_vert(const vvvprop_t &S1,const vvvprop_t &S2,valarray<jvert_t> &jVert)
{
#pragma omp parallel for collapse (4)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int igam=0;igam<16;igam++)
                {
                    if(ntypes==6)
                    {
                        // LO
                        jVert[gbil::LO][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::LO][mr_fw], S2[ijack][qprop::LO][mr_bw],igam);
                        
                        // EM: Self + Tadpole + Exchange
                        jVert[gbil::PH][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::LO][mr_fw],S2[ijack][qprop::FF][mr_bw],igam) +
                            make_vertex(S1[ijack][qprop::LO][mr_fw],S2[ijack][qprop::T ][mr_bw],igam) +
                            make_vertex(S1[ijack][qprop::FF][mr_fw],S2[ijack][qprop::LO][mr_bw],igam) +
                            make_vertex(S1[ijack][qprop::T ][mr_fw],S2[ijack][qprop::LO][mr_bw],igam) +
                            make_vertex(S1[ijack][qprop::F ][mr_fw],S1[ijack][qprop::F ][mr_bw],igam) ;
                        
                        // Pfw
                        jVert[gbil::Pfw][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::P ][mr_fw],S2[ijack][qprop::LO][mr_bw],igam);
                        
                        // Pbw
                        jVert[gbil::Pbw][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::LO][mr_fw],S2[ijack][qprop::P ][mr_bw],igam);
                        
                        // Sfw
                        jVert[gbil::Sfw][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::S ][mr_fw],S2[ijack][qprop::LO][mr_bw],igam);
                        
                        // Sbw
                        jVert[gbil::Sbw][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::LO][mr_fw],S2[ijack][qprop::S ][mr_bw],igam);
                    }
                    if(ntypes==3)
                    {
                        // LO
                        jVert[gbil::LO][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::LO][mr_fw], S2[ijack][qprop::LO][mr_bw],igam);
                        
                        // EM: Self + Tadpole + S + P + Exchange
                        jVert[gbil::PH][ijack][mr_fw][mr_bw][igam] +=
                            make_vertex(S1[ijack][qprop::LO ][mr_fw],S2[ijack][qprop::QED][mr_bw],igam) +
                            make_vertex(S1[ijack][qprop::QED][mr_fw],S2[ijack][qprop::LO ][mr_bw],igam) +
                            make_vertex(S1[ijack][qprop::F  ][mr_fw],S1[ijack][qprop::F  ][mr_bw],igam) ;
                    }
                    
                }
    
}

//project the amputated green function
jproj_t compute_pr_bil( vvvprop_t &jpropOUT_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jpropIN_inv)
{    
    int lambda_size = gbil::nins + 2*jprop::nins - 2;
    
    vector<int> i1;
    vector<int> iv;
    vector<int> i2;
    vector<int> ip;
    
    if(ntypes==6)
    {
        i1={jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,
            jprop::PH,jprop::LO,jprop::P ,jprop::LO,jprop::S ,jprop::LO}; //fw
        iv={gbil::LO,gbil::PH,gbil::Pfw,gbil::Pbw,gbil::Sfw,gbil::Sbw,
            gbil::LO,gbil::LO,gbil::LO ,gbil::LO ,gbil::LO ,gbil::LO };   //vert
        i2={jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,jprop::LO,
            jprop::LO,jprop::PH,jprop::LO,jprop::P ,jprop::LO,jprop::S }; //bw
        
        ip={gbil::LO,gbil::PH,gbil::Pfw,gbil::Pbw,gbil::Sfw,gbil::Sbw,
            gbil::PH,gbil::PH,gbil::Pfw,gbil::Pbw,gbil::Sfw,gbil::Sbw};
    }
    if(ntypes==3)
    {
        i1={jprop::LO,jprop::QED,jprop::LO,jprop::LO};
        iv={gbil::LO,gbil::LO,gbil::LO,gbil::QED};
        i2={jprop::LO,jprop::LO,jprop::QED,jprop::LO};
        
        ip={gbil::LO,gbil::QED,gbil::QED,gbil::QED};
    }
    
    jproj_t pr_bil(vvvvd_t(vvvd_t(vvd_t(vd_t(0.0,nmr),nmr),njacks),nbil),gbil::nins);
    
    const int ibil_of_igam[gbil::nGamma]={0,1,1,1,1,2,3,3,3,3,4,4,4,4,4,4};
    
#pragma omp parallel for collapse(5)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr_fw=0;mr_fw<nmr;mr_fw++)
            for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                for(int k=0;k<lambda_size;k++)
                    for(int igam=0;igam<gbil::nGamma;igam++)
                    {
                        prop_t lambda_igam = jpropOUT_inv[i1[k]][ijack][mr_fw]*
                                             jVert[iv[k]][ijack][mr_fw][mr_bw][igam]*
                                             GAMMA[5]*(jpropIN_inv[i2[k]][ijack][mr_bw]).adjoint()*GAMMA[5];
                        
                        pr_bil[ip[k]][ibil_of_igam[igam]][ijack][mr_fw][mr_bw] +=
                            (lambda_igam*Proj[igam]).trace().real()/12.0;
                    }
    
    return pr_bil;
}

void oper_t::compute_bil()
{
    using namespace gbil;
    
    ifstream jG_data(path_print+"jG");
    
    if(jG_data.good())
    {
        cout<<"Reading bilinears from files: \""<<path_print<<"jG\""<<endl<<endl;
        read_vec_bin(jG,path_print+"jG");
    }
    else
    {
        cout<<"Creating the vertices -- ";
        
        // array of input files to be read in a given conf
        FILE* input[combo];
        
        const vector<string> v_path = setup_read_qprop(input);
        
        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        {
            high_resolution_clock::time_point t0=high_resolution_clock::now();
            
            cout<<endl;
            cout<<"\r\t bilmom = "<<ibilmom+1<<"/"<<_bilmoms<<endl;
            
            const int imom1=bilmoms[ibilmom][1]; // p1
            const int imom2=bilmoms[ibilmom][2]; // p2
            const bool read2=(imom1!=imom2);
            
            // definition of jackknifed propagators
            /* prop1 */
            jprop_t jS1(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            /* prop2 */
            jprop_t jS2(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            
            // definition of jackknifed vertices
            valarray<jvert_t> jVert(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),gbil::nGamma),_nmr),_nmr),njacks),gbil::nins);
            
            cout<<"- Building vertices"<<endl;
            
            double t_span1=0.0, t_span2=0.0, t_span3=0.0;
            
            for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
                for(int ihit=0;ihit<nhits;ihit++)
                {
                    const int mom1=linmoms[imom1][0];
                    const int mom2=linmoms[imom2][0];
                    
                    high_resolution_clock::time_point ta=high_resolution_clock::now();
                    
                    vvvprop_t S1=read_qprop_mom(input,v_path,i_in_clust,ihit,mom1);
                    vvvprop_t S2=(read2)?read_qprop_mom(input,v_path,i_in_clust,ihit,mom2):S1;
                    
                    S1=rotate(S1);
                    S2=(read2)?rotate(S2):S1;
                    
                    high_resolution_clock::time_point tb=high_resolution_clock::now();
                    t_span1 += (duration_cast<duration<double>>(tb-ta)).count();
                    
                    ta=high_resolution_clock::now();
                    
                    build_prop(S1,jS1);
                    if(read2) build_prop(S2,jS2);
                    else {jS2=jS1;}
                    
                    tb=high_resolution_clock::now();
                    t_span2 += (duration_cast<duration<double>>(tb-ta)).count();
                    
                    ta=high_resolution_clock::now();
                    
                    build_vert(S1,S2,jVert);
                    
                    tb=high_resolution_clock::now();
                    t_span3 += (duration_cast<duration<double>>(tb-ta)).count();
                }
            cout<<"\t read: "<<t_span1<<" s"<<endl;
            cout<<"\t build prop: "<<t_span2<<" s"<<endl;
            cout<<"\t build vert: "<<t_span3<<" s"<<endl;
            
            
            cout<<"- Jackknife of propagators and vertices"<<endl;
            
            // jackknife averages
            /* prop1 */
            for(auto &prop1 : jS1) prop1 = jackknife(prop1);
            /* prop2 */
            if(read2)
                for(auto &prop2 : jS2) prop2 = jackknife(prop2);
            else
                jS2=jS1;
            /* vert */
            for(int ins=0; ins<nins; ins++)
                jVert[ins] = jackknife(jVert[ins]);
            
            cout<<"- Inverting propagators"<<endl;
            
            // definition of inverse propagators
            jprop_t jS1_inv(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            jprop_t jS2_inv(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),jprop::nins);
            
            // invert propagators
            /* prop1 */
            jS1_inv[jprop::LO] = invert_jprop(jS1[jprop::LO]);
            for(int i=1;i<jprop::nins;i++)
                jS1_inv[i] = - jS1_inv[jprop::LO]*jS1[i]*jS1_inv[jprop::LO];
            /* prop2 */
            if(read2)
            {
                jS2_inv[jprop::LO] = invert_jprop(jS2[jprop::LO]);
                for(int i=1;i<jprop::nins;i++)
                    jS2_inv[i] = - jS2_inv[jprop::LO]*jS2[i]*jS2_inv[jprop::LO];
            }
            else
                jS2_inv=jS1_inv;
            
            cout<<"- Computing bilinears"<<endl;
            
            // compute the projected green function (S,V,P,A,T)
            jG[ibilmom] = compute_pr_bil(jS1_inv,jVert,jS2_inv);
            
            high_resolution_clock::time_point t1=high_resolution_clock::now();
            duration<double> t_span = duration_cast<duration<double>>(t1-t0);
            cout<<"\t\t time: "<<t_span.count()<<" s"<<endl;
            
        } // close mom loop
        cout<<endl<<endl;
        
        print_vec_bin(jG,path_print+"jG");
    }
}
