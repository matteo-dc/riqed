#include "aliases.hpp"
#include "global.hpp"
#include "read.hpp"
#include "jack.hpp"
#include "Zq.hpp"
#include "Zbil.hpp"
#include "Dirac.hpp" //useless
#include "vertices.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include "fit.hpp"
#include <stdio.h>
#include <stdlib.h>
#include "subtraction.hpp"
#include "evolution.hpp"
#include "print.hpp"
#include "ave_err.hpp"
#include "meslep.hpp"
#include <chrono>

#define EXTERN_OPER

#include "operations.hpp"

#include "vertices.hpp"

using namespace std::chrono;

SCHEME_t get_scheme()
{
    if(scheme=="RI_MOM") return RI_MOM;
    if(scheme=="SMOM") return SMOM;
    
    return ERR;
}

void oper_t::set_moms()
{
    //read mom list
    read_mom_list(path_to_moms);
    moms=mom_list.size();
    cout<<"Read: "<<moms<<" momenta from \""<<mom_path<<"\" (BC: "<<BC<<")."<<endl<<endl;
    
    switch(get_scheme())
    {
        case RI_MOM:
            set_ri_mom_moms();
            break;
        case SMOM:
            cout<<"SMOM!"<<endl;
            set_smom_moms();
            break;
        case ERR:
            cout<<"Invalid scheme."<<endl;
            exit(0);
            break;
    }
    _linmoms=linmoms.size();
    _bilmoms=bilmoms.size();
    _meslepmoms=meslepmoms.size();
//    moms=_linmoms;
    
    print_vec(p2_tilde,path_print+"p2_tilde.txt");
}

void oper_t::set_ri_mom_moms()
{
    linmoms.resize(moms);
    bilmoms.resize(moms);
    meslepmoms.resize(moms);
    
    for(int imom=0;imom<moms;imom++)
        if(filt_moms[imom])
        {
            linmoms[imom]={imom};
            bilmoms[imom]={imom,imom,imom};
            meslepmoms[imom]=bilmoms[imom];
        }
}

void oper_t::set_smom_moms()
{
    // http://xxx.lanl.gov/pdf/0901.2599v2 (Sturm et al.)
    
    linmoms.clear();
    bilmoms.clear();
    
    double eps=1e-10;
    
    // SMOM not yet implemented for 4fermions
    if(compute_4f)
    {
        cout<<" meslepmoms not initialized for SMOM."<<endl;
        exit(0);
    }
    
    for(int i=0;i<moms;i++)
        if(filt_moms[i])
            for(int j=0;j<moms;j++)
                if(filt_moms[j])
                {
                    if(2.0*fabs(p2[i]-p2[j])<(p2[i]+p2[j])*eps)
                    {
                        coords_t momk;
                        
                        p_t k_array, k_tilde_array;
                        double k_sqr=0.0, k_tilde_sqr=0.0;
                        double k_4=0.0, k_tilde_4=0.0;
                        
                        for(size_t mu=0;mu<4;mu++)
                        {
                            momk[mu]=mom_list[i][mu]-mom_list[j][mu];
                            
                            k_array[mu]=2*M_PI*momk[mu]/size[mu];
                            k_sqr+=k_array[mu]*k_array[mu];
                            k_4+=k_array[mu]*k_array[mu]*k_array[mu]*k_array[mu];

                            k_tilde_array[mu]=sin(k_array[mu]);
                            k_tilde_sqr+=k_tilde_array[mu]*k_tilde_array[mu];
                            k_tilde_4+=k_tilde_array[mu]*k_tilde_array[mu]*k_tilde_array[mu]*k_tilde_array[mu];
                        }
                        
                        if(2.0*fabs(p2[i]-k_sqr)<(p2[i]+k_sqr)*eps)
                        {
                            //search in mom_list
                            auto posk = find(mom_list.begin(),mom_list.end(),momk);
                            
                            //if not found, push into mom_list
                            if(posk==mom_list.end())
                            {
                                posk=mom_list.end();
                                
                                mom_list.push_back(momk);
                                p.push_back(k_array);
                                p_tilde.push_back(k_tilde_array);
                                p2.push_back(k_sqr);
                                p2_tilde.push_back(k_tilde_sqr);
                                p4.push_back(k_4);
                                p4_tilde.push_back(k_tilde_4);
                            }
                            
                            const int k=distance(mom_list.begin(),posk);
                            
                            vector<int> pos;
                            
                            //search in the linmoms: if found take the distance, otherwise add
                            for(const int ic : {i,j})
                            {
                                cout<<"searching for "<<ic<<endl;
                                auto pos_ic=find(linmoms.begin(),linmoms.end(),array<int,1>{ic});
                                size_t d;
                                if(pos_ic==linmoms.end())
                                {
                                    //the position will be the end
                                    d=linmoms.size();
                                    //include it
                                    linmoms.push_back({ic});
                                    
                                    cout<<" not found"<<endl;
                                }
                                else
                                {
                                    d=distance(linmoms.begin(),pos_ic);
                                    cout<<" found"<<endl;
                                }
                                
                                //add to the list
                                cout<<"Position: "<<d<<endl;
                                pos.push_back(d);
                            }
                            
                            //store
                            bilmoms.push_back({k,pos[0],pos[1]});
                            
                        } else cout<<"p2-k2 != 0"<<endl;
                    } else cout<<"p1^2-p2^2 != 0"<<endl;
                }
}

////////

void oper_t::create_basic(const int b, const int th, const int msea)
{
//    step = "basic";
    
    _beta=beta[b];
    _beta_label=beta_label[b];
    _nm_Sea=nm_Sea[b];
    _SeaMasses_label=to_string(SeaMasses_label[b][msea]);
    _theta_label=theta_label[th];
    
    path_to_beta = path_ensemble + _beta_label + "_b" + to_string_with_precision(_beta,2) + "/";
    ensamble_name = _beta_label + _SeaMasses_label + _theta_label;
    path_to_ens =  path_to_beta + ensamble_name + "/";
    
    read_input(path_to_ens,ensamble_name);
    path_to_moms = path_to_ens + mom_path;
    
    path_print = path_to_ens+"print/";
    
    _nm=nm;
    _nr=nr;
    _nmr=_nm*_nr;
    
    g2=6.0/_beta;
    g2_tilde=g2/plaquette;
    
    set_moms();
    
    allocate();
    
    deltam_cr=read_deltam_cr(path_to_ens+"deltam_cr_array");
    if(UseEffMass) eff_mass=read_eff_mass(path_to_ens+"eff_mass_array");
    if(UseEffMass and _nm_Sea>1) eff_mass_sea=read_eff_mass_sea(path_to_ens+"eff_mass_sea_array");

    ifstream jZq_data(path_print+"jZq");
    ifstream jZq_em_data(path_print+"jZq_em");
    ifstream jG_0_data(path_print+"jG_0");
    ifstream jG_em_data(path_print+"jG_em");
    ifstream jG_0_4f_data(path_print+"jG_0_4f");
    ifstream jG_em_4f_data(path_print+"jG_em_4f");
    ifstream jpr_meslep_0_data(path_print+"jpr_meslep_0");
    ifstream jpr_meslep_em_data(path_print+"jpr_meslep_em");
    if(jZq_data.good() and jZq_em_data.good() and jG_0_data.good() and jG_em_data.good() and jG_0_4f_data.good() and jG_em_4f_data.good() and jpr_meslep_0_data.good() and jpr_meslep_em_data.good())
    {
        cout<<"Reading data from files"<<endl<<endl;

//        vector<int> Np(_linmoms);

        read_vec_bin(jZq,path_print+"jZq");
        read_vec_bin(jZq_em,path_print+"jZq_em");
        read_vec_bin(jG_0,path_print+"jG_0");
        read_vec_bin(jG_em,path_print+"jG_em");
        read_vec_bin(jG_0_4f,path_print+"jG_0_4f");
        read_vec_bin(jG_em_4f,path_print+"jG_em_4f");
        read_vec_bin(jpr_meslep_0,path_print+"jpr_meslep_0");
        read_vec_bin(jpr_meslep_em,path_print+"jpr_meslep_em");

//        READ_BIN(jZq);
//        READ_BIN(jZq_em);
////        READ_BIN(Np);
//        READ_BIN(jG_0);
//        READ_BIN(jG_em);
        
    }
    else
    {
        switch(get_scheme())
        {
            case RI_MOM:
                ri_mom();
                break;
            case SMOM:
                cout<<"SMOM!"<<endl;
                smom();
                break;
            case ERR:
                cout<<"Invalid scheme."<<endl;
                exit(0);
                break;
        }
    }
    
    compute_Zbil();
    if(compute_4f) compute_Z4f();
    
}

void oper_t::ri_mom()
{
    compute_prop();
    compute_bil();
    if(compute_4f) compute_meslep();
}

void oper_t::smom()
{
    ri_mom();
}


//////////

void oper_t::allocate()
{
    jZq.resize(_linmoms);
    jZq_em.resize(_linmoms);
    
    jG_0.resize(_bilmoms);
    jG_em.resize(_bilmoms);
    
    jZ.resize(_bilmoms);
    jZ_em.resize(_bilmoms);
    
    jG_0_4f.resize(_bilmoms);
    jG_em_4f.resize(_bilmoms);
    
    jpr_meslep_0.resize(_bilmoms);
    jpr_meslep_em.resize(_bilmoms);
    
    jZ_4f.resize(_bilmoms);
    jZ_em_4f.resize(_bilmoms);
    
    for(auto &ijack : jZq)
    {
        ijack.resize(njacks);
        for(auto &mr : ijack)
            mr.resize(_nmr);
    }
    
    for(auto &ijack : jZq_em)
    {
        ijack.resize(njacks);
        for(auto &mr : ijack)
            mr.resize(_nmr);
    }
    
    
    for(auto &ibil : jG_0)
    {
        ibil.resize(nbil);
        for(auto &ijack : ibil)
        {
            ijack.resize(njacks);
            for(auto &mr1 : ijack)
            {
                mr1.resize(_nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(_nmr);
            }
        }
    }
    
    for(auto &ibil : jG_em)
    {
        ibil.resize(nbil);
        for(auto &ijack : ibil)
        {
            ijack.resize(njacks);
            for(auto &mr1 : ijack)
            {
                mr1.resize(_nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(_nmr);
            }
        }
    }
    
    for(auto &ibil : jZ)
    {
        ibil.resize(nbil);
        for(auto &ijack : ibil)
        {
            ijack.resize(njacks);
            for(auto &mr1 : ijack)
            {
                mr1.resize(_nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(_nmr);
            }
        }
    }
    
    for(auto &ibil : jZ_em)
    {
        ibil.resize(nbil);
        for(auto &ijack : ibil)
        {
            ijack.resize(njacks);
            for(auto &mr1 : ijack)
            {
                mr1.resize(_nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(_nmr);
            }
        }
    }

    for(auto &ibil : jG_0_4f)
    {
        ibil.resize(nbil+1);
        for(auto &ijack : ibil)
        {
            ijack.resize(njacks);
            for(auto &mr1 : ijack)
            {
                mr1.resize(_nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(_nmr);
            }
        }
    }
    
    for(auto &ibil : jG_em_4f)
    {
        ibil.resize(nbil+1);
        for(auto &ijack : ibil)
        {
            ijack.resize(njacks);
            for(auto &mr1 : ijack)
            {
                mr1.resize(_nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(_nmr);
            }
        }
    }

    for(auto &iop1 : jpr_meslep_0)
    {
        iop1.resize(nbil);
        for(auto &iop2 : iop1)
        {
            iop2.resize(nbil);
            for(auto &ijack : iop2)
            {
                ijack.resize(njacks);
                for(auto &mr1 : ijack)
                {
                    mr1.resize(_nmr);
                    for(auto &mr2 : mr1)
                        mr2.resize(_nmr);
                }
            }
        }
    }
    
    for(auto &iop1 : jpr_meslep_em)
    {
        iop1.resize(nbil);
        for(auto &iop2 : iop1)
        {
            iop2.resize(nbil);
            for(auto &ijack : iop2)
            {
                ijack.resize(njacks);
                for(auto &mr1 : ijack)
                {
                    mr1.resize(_nmr);
                    for(auto &mr2 : mr1)
                        mr2.resize(_nmr);
                }
            }
        }
    }
    
    for(auto &iop1 : jZ_4f)
    {
        iop1.resize(nbil);
        for(auto &iop2 : iop1)
        {
            iop2.resize(nbil);
            for(auto &ijack : iop2)
            {
                ijack.resize(njacks);
                for(auto &mr1 : ijack)
                {
                    mr1.resize(_nmr);
                    for(auto &mr2 : mr1)
                        mr2.resize(_nmr);
                }
                
            }
        }
    }
    
    for(auto &iop1 : jZ_em_4f)
    {
        iop1.resize(nbil);
        for(auto &iop2 : iop1)
        {
            iop2.resize(nbil);
            for(auto &ijack : iop2)
            {
                ijack.resize(njacks);
                for(auto &mr1 : ijack)
                {
                    mr1.resize(_nmr);
                    for(auto &mr2 : mr1)
                        mr2.resize(_nmr);
                }
                
            }
        }
    }
    

}

void oper_t::resize_output(oper_t out)
{
    (out.jZq).resize(out._linmoms);
    (out.jZq_em).resize(out._linmoms);
    
    (out.jG_0).resize(out._bilmoms);
    (out.jG_em).resize(out._bilmoms);
    
    (out.jZ).resize(out._bilmoms);
    (out.jZ_em).resize(out._bilmoms);
    
    for(auto &ijack : out.jZq)
        for(auto &mr : ijack)
            mr.resize(out._nmr);
    
    for(auto &ijack : out.jZq_em)
        for(auto &mr : ijack)
            mr.resize(out._nmr);
    
    for(auto &ibil : out.jG_0)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }
    
    for(auto &ibil : out.jG_em)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }
    
    for(auto &ibil : out.jZ)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }
    
    for(auto &ibil : out.jZ_em)
        for(auto &ijack : ibil)
            for(auto &mr1 : ijack)
            {
                mr1.resize(out._nmr);
                for(auto &mr2 : mr1)
                    mr2.resize(out._nmr);
            }
}

void build_prop(const vvvprop_t &S, jprop_t &jS_0,jprop_t &jS_self_tad,jprop_t &jS_P)
{
    vvvprop_t S_LO_EM_P(vvprop_t(vprop_t(prop_t::Zero(),nmr),njacks),3);
    
#pragma omp parallel for collapse(3)
    for(int m=0;m<nm;m++)
        for(int r=0;r<nr;r++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                int mr = r + nr*m;
                
                S_LO_EM_P[LO][ijack][mr] = S[ijack][0][mr];  // Leading order
            
//                // Electromagnetic correction:  S_em = S_self + S_tad -+ deltam_cr*S_P
//                if(r==0) S_LO_and_EM[EM][ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] + deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=0
//                if(r==1) S_LO_and_EM[EM][ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr] - deltam_cr[ijack][m][m]*S[ijack][4][mr]; //r=1

                S_LO_EM_P[EM][ijack][mr] = S[ijack][2][mr] + S[ijack][3][mr];  // self energy + tadpole
                
                // here the tau3 matrix elements are explicit: diag(tau3)=(+1,-1)
                if(r==0) S_LO_EM_P[P][ijack][mr] = +1.0*S[ijack][4][mr];
                if(r==1) S_LO_EM_P[P][ijack][mr] = -1.0*S[ijack][4][mr];
                
                
                jS_0[ijack][mr] += S_LO_EM_P[LO][ijack][mr];
                jS_self_tad[ijack][mr] += S_LO_EM_P[EM][ijack][mr];
                jS_P[ijack][mr] += S_LO_EM_P[P][ijack][mr];
            }
}

void oper_t::compute_prop()
{
    cout<<"Creating the propagators -- ";

    // array of input files to be read in a given conf
    FILE* input[combo];
    vector<string> v_path = setup_read_qprop(input);

    vvvd_t jZq_LO_and_EM(vvd_t(vd_t(0.0,_nmr),njacks),2);
    
    for(int ilinmom=0; ilinmom<_linmoms; ilinmom++)
    {
        cout<<"\r\t linmom = "<<ilinmom+1<<"/"<<_linmoms<<endl;
        
        // initialize propagators
//        vvvprop_t S_LO_EM_P(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),3);  // size=3 > {LO,EM,P}
        
        // definition of jackknifed propagators
        jprop_t jS_0(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS_self_tad(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS_em(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        
        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
            for(int ihit=0;ihit<nhits;ihit++)
            {
                const vvvprop_t S=read_qprop_mom(input,v_path,i_in_clust,ihit,ilinmom);
                build_prop(S,jS_0,jS_self_tad,jS_P);
            }
        
        // jackknife average
        jS_0 = jackknife(jS_0);
        jS_self_tad = jackknife(jS_self_tad);
        jS_P = jackknife(jS_P);
        
        // build the complete electromagnetic correction
#pragma omp parallel for collapse(3)
        for(int m=0;m<nm;m++)
            for(int r=0;r<nr;r++)
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    int mr = r + nr*m;
                    jS_em[ijack][mr] = jS_self_tad[ijack][mr] + deltam_cr[ijack][m][m]*jS_P[ijack][mr];
                }
        
        // invert propagator
        vvvprop_t jS_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
        
        jS_inv_LO_and_EM[LO] = invert_jprop(jS_0);
        jS_inv_LO_and_EM[EM] = jS_inv_LO_and_EM[LO]*jS_em*jS_inv_LO_and_EM[LO];
        
        // compute quark field RCs (Zq or Sigma1 established from input file!) and store
        jZq_LO_and_EM = compute_jZq(jS_inv_LO_and_EM,ilinmom);
        
        jZq[ilinmom] = jZq_LO_and_EM[LO];
        jZq_em[ilinmom] = - jZq_LO_and_EM[EM];
        
    } // close linmoms loop
    
    print_vec_bin(jZq,path_print+"jZq");
    print_vec_bin(jZq_em,path_print+"jZq_em");
}

void oper_t::compute_bil()
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
        jprop_t jS1_0(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS1_self_tad(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS1_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS1_em(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        /* prop2 */
        jprop_t jS2_0(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS2_self_tad(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS2_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS2_em(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        
        // definition of jackknifed vertices
        valarray<jvert_t> jVert_LO_EM_P(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),_nmr),_nmr),njacks),4);  // size=4 > {LO,self+tadpole,P(fw),P(bw)}
        valarray<jvert_t> jVert_LO_and_EM(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),_nmr),_nmr),njacks),2);
        
        cout<<"- Building vertices"<<endl;
        
        double t_span1=0.0, t_span2=0.0, t_span3=0.0;
        
        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
            for(int ihit=0;ihit<nhits;ihit++)
            {
                const int mom1=linmoms[imom1][0];
                const int mom2=linmoms[imom2][0];
                
                high_resolution_clock::time_point ta=high_resolution_clock::now();
                
                const vvvprop_t S1=read_qprop_mom(input,v_path,i_in_clust,ihit,mom1);
                const vvvprop_t S2=(read2)?read_qprop_mom(input,v_path,i_in_clust,ihit,mom2):S1;
                
                high_resolution_clock::time_point tb=high_resolution_clock::now();
                t_span1 += (duration_cast<duration<double>>(tb-ta)).count();
                
                ta=high_resolution_clock::now();
                
                build_prop(S1,jS1_0,jS1_self_tad,jS1_P);
                if(read2) build_prop(S2,jS2_0,jS2_self_tad,jS2_P);
                else {jS2_0=jS1_0; jS2_self_tad=jS1_self_tad ; jS2_P=jS1_P;}
                
                tb=high_resolution_clock::now();
                t_span2 += (duration_cast<duration<double>>(tb-ta)).count();
                
                ta=high_resolution_clock::now();

                build_vert(S1,S2,jVert_LO_EM_P);
                
                tb=high_resolution_clock::now();
                t_span3 += (duration_cast<duration<double>>(tb-ta)).count();
            }
        cout<<"\t read: "<<t_span1<<" s"<<endl;
        cout<<"\t build prop: "<<t_span2<<" s"<<endl;
        cout<<"\t build vert: "<<t_span3<<" s"<<endl;

    
        cout<<"- Jackknife of propagators and vertices"<<endl;
        
        // jackknife averages
        jS1_0=jackknife(jS1_0);
        jS1_self_tad = jackknife(jS1_self_tad);
        jS1_P=jackknife(jS1_P);

        jS2_0=(read2)?jackknife(jS2_0):jS1_0;
        jS2_self_tad=(read2)?jackknife(jS2_self_tad):jS1_self_tad;
        jS2_P=(read2)?jackknife(jS2_P):jS1_P;
        
        jVert_LO_EM_P[LO]=jackknife(jVert_LO_EM_P[LO]);
        jVert_LO_EM_P[EM]=jackknife(jVert_LO_EM_P[EM]);
        jVert_LO_EM_P[2]=jackknife(jVert_LO_EM_P[2]); // fw
        jVert_LO_EM_P[3]=jackknife(jVert_LO_EM_P[3]); // bw
        
        // build the complete electromagnetic correction
#pragma omp parallel for collapse(3)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m=0;m<nm;m++)
                for(int r=0;r<nr;r++)
                {
                    int mr = r + nr*m;
                    
                    jS1_em[ijack][mr]=jS1_self_tad[ijack][mr] + deltam_cr[ijack][m][m]*jS1_P[ijack][mr];
                    (read2)?jS2_em[ijack][mr]=jS2_self_tad[ijack][mr] + deltam_cr[ijack][m][m]*jS2_P[ijack][mr]:jS2_em[ijack][mr]=jS1_em[ijack][mr];
                }
        
#pragma omp parallel for collapse (4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                    for(int igam=0;igam<16;igam++)
                    {
                        int r_fw = mr_fw%nr;
                        int m_fw = (mr_fw-r_fw)/nr;
                        int r_bw = mr_bw%nr;
                        int m_bw = (mr_bw-r_bw)/nr;
                        
                        jVert_LO_and_EM[LO][ijack][mr_fw][mr_bw][igam] = jVert_LO_EM_P[LO][ijack][mr_fw][mr_bw][igam];
                        
                        jVert_LO_and_EM[EM][ijack][mr_fw][mr_bw][igam] = jVert_LO_EM_P[EM][ijack][mr_fw][mr_bw][igam] + deltam_cr[ijack][m_fw][m_fw]*jVert_LO_EM_P[2][ijack][mr_fw][mr_bw][igam] + deltam_cr[ijack][m_bw][m_bw]*jVert_LO_EM_P[3][ijack][mr_fw][mr_bw][igam];
                    }
        
        
        cout<<"- Inverting propagators"<<endl;

        // invert propagators
        vvvprop_t jS1_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
        vvvprop_t jS2_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
        
        jS1_inv_LO_and_EM[LO] = invert_jprop(jS1_0);
        jS1_inv_LO_and_EM[EM] = jS1_inv_LO_and_EM[LO]*jS1_em*jS1_inv_LO_and_EM[LO];
        jS2_inv_LO_and_EM[LO] = (read2)?invert_jprop(jS2_0):jS1_inv_LO_and_EM[LO];
        jS2_inv_LO_and_EM[EM] = (read2)?jS2_inv_LO_and_EM[LO]*jS2_em*jS2_inv_LO_and_EM[LO]:jS1_inv_LO_and_EM[EM];
        
//        cout<<"- Computing Zq"<<endl;
//        
//        // compute Zq relative to imom1 and eventually to imom2
//        vvvd_t jZq1_LO_and_EM = compute_jZq(jS1_inv_LO_and_EM,imom1);
//        vvvd_t jZq2_LO_and_EM = (read2)?compute_jZq(jS2_inv_LO_and_EM,imom2):jZq1_LO_and_EM;
//        
//        jZq[imom1] = jZq1_LO_and_EM[LO];
//        jZq_em[imom1] = - jZq1_LO_and_EM[EM];
//        
//        if(read2)
//        {
//            jZq[imom2] = jZq2_LO_and_EM[LO];
//            jZq_em[imom2] = - jZq2_LO_and_EM[EM];
//        }
        
        cout<<"- Computing bilinears"<<endl;
        
        // compute the projected green function (S,V,P,A,T)
        vvvvvd_t jG_LO_and_EM = compute_pr_bil(jS1_inv_LO_and_EM,jVert_LO_and_EM,jS2_inv_LO_and_EM);
        
        jG_0[ibilmom] = jG_LO_and_EM[LO];
        jG_em[ibilmom] = jG_LO_and_EM[EM];
        
        high_resolution_clock::time_point t1=high_resolution_clock::now();
        duration<double> t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"\t\t time: "<<t_span.count()<<" s"<<endl;
        
    } // close mom loop
    cout<<endl<<endl;
    
    print_vec_bin(jG_0,path_print+"jG_0");
    print_vec_bin(jG_em,path_print+"jG_em");
}

void oper_t::compute_Zbil()
{
    Zbil_computed=true;
    
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        //compute Z's according to 'riqed.pdf', one for each momentum
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
                    for(int ibil=0;ibil<nbil;ibil++)
                    {
                        jZ[ibilmom][ibil][ijack][mr_fw][mr_bw] = sqrt(jZq[imom1][ijack][mr_fw]*jZq[imom2][ijack][mr_bw])/jG_0[ibilmom][ibil][ijack][mr_fw][mr_bw];
                        
                        jZ_em[ibilmom][ibil][ijack][mr_fw][mr_bw] = - jG_em[ibilmom][ibil][ijack][mr_fw][mr_bw]/jG_0[ibilmom][ibil][ijack][mr_fw][mr_bw] + 0.5*(jZq_em[imom1][ijack][mr_fw]/jZq[imom1][ijack][mr_fw] + jZq_em[imom2][ijack][mr_bw]/jZq[imom2][ijack][mr_bw]);
                    }
        
    }// close mom loop
}


void oper_t::compute_meslep()
{
    cout<<"Creating the vertices -- ";
    
    //these are the charges in the lagrangian
    const double ql=-1.0;     //!< the program simulates muon *particle*
    const double q1=-1.0/3.0; //!< charge of the quark1
    const double q2=+2.0/3.0; //!< charge of the quark2
    
    // array of input files to be read in a given conf
    FILE* input_q[combo];
    FILE* input_l[combo_lep];

    const vector<string> v_path_q = setup_read_qprop(input_q);
    const vector<string> v_path_l = setup_read_lprop(input_l);
    
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        high_resolution_clock::time_point t0=high_resolution_clock::now();
        
        cout<<endl;
        cout<<"\r\t meslepmom = "<<ibilmom+1<<"/"<<_bilmoms<<endl;
        
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        const bool read2=(imom1!=imom2);
        
        // definition of jackknifed propagators
        /* prop1 */
        jprop_t jS1_0(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS1_self_tad(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS1_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS1_em(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        /* prop2 */
        jprop_t jS2_0(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS2_self_tad(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS2_P(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        jprop_t jS2_em(valarray<prop_t>(prop_t::Zero(),_nmr),njacks);
        
        // definition of jackknifed vertices
        valarray<jvert_t> jVert_LO_EM_P(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),20),_nmr),_nmr),njacks),4);  // size=4 > {LO,self+tadpole,P(fw),P(bw)}
        valarray<jvert_t> jVert_LO_and_EM(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),20),_nmr),_nmr),njacks),2);
        
        // definition of jackknifed meslep ("in" & "out" diagrams)
        valarray<jmeslep_t> jmeslep_QCD_IN_OUT(jmeslep_t(jvert_t(vvvprop_t(vvprop_t(vprop_t(prop_t::Zero(),16),16),_nmr),_nmr),njacks),3);
        
        cout<<"- Building vertices"<<endl;
        
        double t_span1=0.0, t_span2=0.0, t_span3=0.0, t_span4=0.0;
        
        for(int i_in_clust=0;i_in_clust<clust_size;i_in_clust++)
            for(int ihit=0;ihit<nhits;ihit++)
            {
                const int mom1=linmoms[imom1][0];
                const int mom2=linmoms[imom2][0];
                const int momlep=mom1;
                
                high_resolution_clock::time_point ta=high_resolution_clock::now();
                
                const vvvprop_t S1=read_qprop_mom(input_q,v_path_q,i_in_clust,ihit,mom1);
                const vvvprop_t S2=(read2)?read_qprop_mom(input_q,v_path_q,i_in_clust,ihit,mom2):S1;
                const vvprop_t L=read_lprop_mom(input_l,v_path_l,i_in_clust,ihit,momlep);
                
                high_resolution_clock::time_point tb=high_resolution_clock::now();
                t_span1 += (duration_cast<duration<double>>(tb-ta)).count();
                
                ta=high_resolution_clock::now();
                
                build_prop(S1,jS1_0,jS1_self_tad,jS1_P);
                if(read2) build_prop(S2,jS2_0,jS2_self_tad,jS2_P);
                else {jS2_0=jS1_0; jS2_self_tad=jS1_self_tad ; jS2_P=jS1_P;}
                
                tb=high_resolution_clock::now();
                t_span2 += (duration_cast<duration<double>>(tb-ta)).count();
                
                ta=high_resolution_clock::now();
                
                build_vert_4f(S1,S2,jVert_LO_EM_P,q1,q2);
                
                tb=high_resolution_clock::now();
                t_span3 += (duration_cast<duration<double>>(tb-ta)).count();
                
                ta=high_resolution_clock::now();
                
                build_meslep(S1,S2,L,jmeslep_QCD_IN_OUT);
                
                tb=high_resolution_clock::now();
                t_span4 += (duration_cast<duration<double>>(tb-ta)).count();
                
            }
        cout<<"\t read: "<<t_span1<<" s"<<endl;
        cout<<"\t build prop: "<<t_span2<<" s"<<endl;
        cout<<"\t build vert: "<<t_span3<<" s"<<endl;
        cout<<"\t build meslep: "<<t_span4<<" s"<<endl;
        
        
        cout<<"- Jackknife of propagators, vertices and meslep"<<endl;
        
        // jackknife averages
        jS1_0=jackknife(jS1_0);
        jS1_self_tad = jackknife(jS1_self_tad);
        jS1_P=jackknife(jS1_P);
        
        jS2_0=(read2)?jackknife(jS2_0):jS1_0;
        jS2_self_tad=(read2)?jackknife(jS2_self_tad):jS1_self_tad;
        jS2_P=(read2)?jackknife(jS2_P):jS1_P;
        
        jVert_LO_EM_P[LO]=jackknife(jVert_LO_EM_P[LO]);
        jVert_LO_EM_P[EM]=jackknife(jVert_LO_EM_P[EM]);
        jVert_LO_EM_P[2]=jackknife(jVert_LO_EM_P[2]); // fw
        jVert_LO_EM_P[3]=jackknife(jVert_LO_EM_P[3]); // bw
        
        jmeslep_QCD_IN_OUT[QCD]=jackknife(jmeslep_QCD_IN_OUT[QCD]);
        jmeslep_QCD_IN_OUT[IN]=jackknife(jmeslep_QCD_IN_OUT[IN]);
        jmeslep_QCD_IN_OUT[OUT]=jackknife(jmeslep_QCD_IN_OUT[OUT]);
        
        
        // build the complete electromagnetic correction
#pragma omp parallel for collapse(3)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int m=0;m<nm;m++)
                for(int r=0;r<nr;r++)
                {
                    int mr = r + nr*m;
                    
                    jS1_em[ijack][mr]=jS1_self_tad[ijack][mr] + deltam_cr[ijack][m][m]*jS1_P[ijack][mr];
                    (read2)?jS2_em[ijack][mr]=jS2_self_tad[ijack][mr] + deltam_cr[ijack][m][m]*jS2_P[ijack][mr]:jS2_em[ijack][mr]=jS1_em[ijack][mr];
                }
        
#pragma omp parallel for collapse (4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<nmr;mr_bw++)
                    for(int igam=0;igam<20;igam++)
                    {
                        int r_fw = mr_fw%nr;
                        int m_fw = (mr_fw-r_fw)/nr;
                        int r_bw = mr_bw%nr;
                        int m_bw = (mr_bw-r_bw)/nr;
                        
                        jVert_LO_and_EM[LO][ijack][mr_fw][mr_bw][igam] = jVert_LO_EM_P[LO][ijack][mr_fw][mr_bw][igam];
                        
                        jVert_LO_and_EM[EM][ijack][mr_fw][mr_bw][igam] = jVert_LO_EM_P[EM][ijack][mr_fw][mr_bw][igam] + deltam_cr[ijack][m_fw][m_fw]*jVert_LO_EM_P[2][ijack][mr_fw][mr_bw][igam] + deltam_cr[ijack][m_bw][m_bw]*jVert_LO_EM_P[3][ijack][mr_fw][mr_bw][igam];
                    }

        cout<<"- Inverting propagators"<<endl;
        
        // invert propagators
        vvvprop_t jS1_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
        vvvprop_t jS2_inv_LO_and_EM(vvprop_t(vprop_t(prop_t::Zero(),_nmr),njacks),2);
        
        jS1_inv_LO_and_EM[LO] = invert_jprop(jS1_0);
        jS1_inv_LO_and_EM[EM] = jS1_inv_LO_and_EM[LO]*jS1_em*jS1_inv_LO_and_EM[LO];
        jS2_inv_LO_and_EM[LO] = (read2)?invert_jprop(jS2_0):jS1_inv_LO_and_EM[LO];
        jS2_inv_LO_and_EM[EM] = (read2)?jS2_inv_LO_and_EM[LO]*jS2_em*jS2_inv_LO_and_EM[LO]:jS1_inv_LO_and_EM[EM];
        
        cout<<"- Computing bilinears"<<endl;
        
        // compute the projected green function (S,V,P,A,T)
        vvvvvd_t jG_LO_and_EM = compute_pr_bil_4f(jS1_inv_LO_and_EM,jVert_LO_and_EM,jS2_inv_LO_and_EM,q1,q2);
        
        jG_0_4f[ibilmom] = jG_LO_and_EM[LO];
        jG_em_4f[ibilmom] = jG_LO_and_EM[EM];
        
        cout<<"- Computing projected meslep"<<endl;
        
        jvproj_meslep_t jpr_meslep_QCD_IN_OUT = compute_pr_meslep(jS1_inv_LO_and_EM,jmeslep_QCD_IN_OUT,jS2_inv_LO_and_EM,q1,q2,ql);
        
        jpr_meslep_0[ibilmom] = jpr_meslep_QCD_IN_OUT[QCD];
        jpr_meslep_em[ibilmom] = jpr_meslep_QCD_IN_OUT[IN] + jpr_meslep_QCD_IN_OUT[OUT];

        high_resolution_clock::time_point t1=high_resolution_clock::now();
        duration<double> t_span = duration_cast<duration<double>>(t1-t0);
        cout<<"\t\t time: "<<t_span.count()<<" s"<<endl;
        
    } // close mom loop
    cout<<endl<<endl;
    
    print_vec_bin(jG_0_4f,path_print+"jG_0_4f");
    print_vec_bin(jG_em_4f,path_print+"jG_em_4f");
    print_vec_bin(jpr_meslep_0,path_print+"jpr_meslep_0");
    print_vec_bin(jpr_meslep_em,path_print+"jpr_meslep_em");
}

void oper_t::compute_Z4f()
{
    vector<vector<int>> ibil_of_iop = {{0,2},{0,2},{1,3},{1,3},{4,5}};
    
    //these are the charges in the lagrangian
    const double q1=-1.0/3.0; //!< charge of the quark1
    const double q2=+2.0/3.0; //!< charge of the quark2
    
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        //compute Z's according to 'riqed.pdf', one for each momentum
#pragma omp parallel for collapse(4)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr_fw=0;mr_fw<_nmr;mr_fw++)
                for(int mr_bw=0;mr_bw<_nmr;mr_bw++)
                    for(int iop1=0;iop1<nbil;iop1++)
                        for(int iop2=0;iop2<nbil;iop2++)
                        {
                            //                        jZ_4f[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw] = sqrt(jZq[imom1][ijack][mr_fw]*jZq[imom2][ijack][mr_bw])/jG_0[ibilmom][ibil][ijack][mr_fw][mr_bw];
                            
                            jZ_em_4f[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw] = - jpr_meslep_em[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw]/jpr_meslep_0[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw];
                            
                            if(iop1==iop2)
                            {
                                int ibil1=ibil_of_iop[iop1][0];
                                int ibil2=ibil_of_iop[iop1][1];
                                
                                jZ_em_4f[ibilmom][iop1][iop2][ijack][mr_fw][mr_bw] += - (jG_em_4f[ibilmom][ibil1][ijack][mr_fw][mr_bw] + jG_em_4f[ibilmom][ibil2][ijack][mr_fw][mr_bw])/(jG_0_4f[ibilmom][ibil1][ijack][mr_fw][mr_bw] + jG_0_4f[ibilmom][ibil2][ijack][mr_fw][mr_bw]) + 0.5*(q1*q1*jZq_em[imom1][ijack][mr_fw]/jZq[imom1][ijack][mr_fw] + q2*q2*jZq_em[imom2][ijack][mr_bw]/jZq[imom2][ijack][mr_bw]);
                            }
                            
                            if(mr_bw==0 and mr_fw==0)
                            {
                                if(iop1==0 and iop2==0) cout<<"ibilmom "<<ibilmom<<" ijack "<<ijack<<endl;
                                cout<<jZ_em_4f[ibilmom][iop1][iop2][ijack][0][0]<<" \t";
                                if(iop2==nbil-1) cout<<endl;
                                if(iop1==nbil-1 and iop2==nbil-1) cout<<endl;
                            }
                        }
        
    }// close mom loop

}






oper_t oper_t::average_r(/*const bool recompute_Zbil*/)
{
    cout<<"Averaging over r"<<endl<<endl;
    
    oper_t out=(*this);
    
    out._nr=1;
    out._nm=_nm;
    out._nmr=(out._nm)*(out._nr);
    
    out.allocate();
    
    if(UseEffMass==1)
    {
        vvvd_t eff_mass_temp(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks);
        
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mA=0; mA<_nm; mA++)
                for(int mB=0; mB<_nm; mB++)
                    for(int r=0; r<_nr; r++)
                    {
                        eff_mass_temp[ijack][mA][mB] += eff_mass[ijack][r+_nr*mA][r+_nr*mB]/_nr;
                    }
        
        out.eff_mass=eff_mass_temp;
        
        
        if(_nm_Sea>1)
        {
            vvvd_t eff_mass_sea_temp(vvd_t(vd_t(0.0,out._nr),out._nr),njacks);
            
            for(int ijack=0;ijack<njacks;ijack++)
                for(int r=0; r<_nr; r++)
                    eff_mass_sea_temp[ijack][0][0] += eff_mass[ijack][r][r]/_nr;
            
            out.eff_mass_sea=eff_mass_sea_temp;
        }
    }
    
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
    {
        vvd_t jZq_mom_temp(vd_t(0.0,out._nmr),njacks);
        vvd_t jZq_em_mom_temp(vd_t(0.0,out._nmr),njacks);
        
        for(int m=0; m<_nm; m++)
            for(int r=0; r<_nr; r++)
            {
                //LO
                for(int ijack=0;ijack<njacks;ijack++) jZq_mom_temp[ijack][m] += jZq[ilinmom][ijack][r+_nr*m]/_nr;
                //EM
                for(int ijack=0;ijack<njacks;ijack++) jZq_em_mom_temp[ijack][m] += jZq_em[ilinmom][ijack][r+_nr*m]/_nr;
            }
        
        (out.jZq)[ilinmom] = jZq_mom_temp;
        (out.jZq_em)[ilinmom] = jZq_em_mom_temp;
        
    }
    
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        jproj_t jG_0_mom_temp(vvvd_t(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks),nbil);
        jproj_t jG_em_mom_temp(vvvd_t(vvd_t(vd_t(0.0,out._nmr),out._nmr),njacks),nbil);
        
        for(int mA=0; mA<_nm; mA++)
            for(int mB=0; mB<_nm; mB++)
                for(int r=0; r<_nr; r++)
                {
                    //LO
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int ibil=0; ibil<5; ibil++)
                            jG_0_mom_temp[ibil][ijack][mA][mB] += jG_0[ibilmom][ibil][ijack][r+_nr*mA][r+_nr*mB]/_nr;
                    //EM
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int ibil=0; ibil<5; ibil++)
                            jG_em_mom_temp[ibil][ijack][mA][mB] += jG_em[ibilmom][ibil][ijack][r+nr*mA][r+nr*mB]/nr;
                }
        
        (out.jG_0)[ibilmom]=jG_0_mom_temp;
        (out.jG_em)[ibilmom]=jG_em_mom_temp;
    }
    
    out.compute_Zbil();
    
    
    // ADD 4fermions quantities!!!
    
    return out;
}
    
oper_t oper_t::chiral_extr()
{
    cout<<"Chiral extrapolation"<<endl<<endl;
    
    oper_t out=(*this);
    
    out._nr=_nr;
    out._nm=1;
    out._nmr=(out._nm)*(out._nr);
    
//    resize_output(out);
    out.allocate();
    
    vvvvd_t G_0_err = get<1>(ave_err(jG_0));    //[imom][ibil][mr1][mr2]
    vvvvd_t G_em_err = get<1>(ave_err(jG_em));
    
    vvd_t Zq_err = get<1>(ave_err(jZq));        //[imom][mr]
    vvd_t Zq_em_err = get<1>(ave_err(jZq_em));
    
    //Sum of quark masses for the extrapolation
//    vd_t mass_sum(0.0,10);
//    int i_sum = 0;
//    for (int i=0; i<nm; i++)
//        for(int j=i;j<nm;j++)
//        {
//            mass_sum[i_sum] = mass_val[i]+mass_val[j];
//            i_sum++;
//        }

    // average of eff_mass
    vvd_t M_eff = get<0>(ave_err(eff_mass));
    
    //range for fit Zq
    int x_min_q=0;
    int x_max_q=_nm-1;
    
    // range for fit bilinears
    int x_min=0;
    int x_max=_nm*(_nm+1)/2-1;
    
    // number of fit parameters for bilinears
    int npar[5]={3,2,3,2,2};
    
    //extrapolate Zq
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
    {
        for(int r=0; r<_nr; r++)
        {
            vvd_t coord_q(vd_t(0.0,_nm),2); // coords at fixed r
            
            vvvd_t jZq_r(vvd_t(vd_t(0.0,_nm),njacks),_linmoms);
            vvvd_t jZq_em_r(vvd_t(vd_t(0.0,_nm),njacks),_linmoms);
            
            vvd_t Zq_err_r(vd_t(0.0,_nm),_linmoms);
            vvd_t Zq_em_err_r(vd_t(0.0,_nm),_linmoms);
            
            for(int m=0; m<_nm; m++)
            {
                int mr = r + _nr*m;
                
                coord_q[0][m] = 1.0;
                if(UseEffMass==0)
                    coord_q[1][m]= mass_val[m];
                else if(UseEffMass==0)
                    coord_q[1][m] = pow(M_eff[mr][mr],2.0);
                
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    jZq_r[ilinmom][ijack][m]=jZq[ilinmom][ijack][mr];
                    jZq_em_r[ilinmom][ijack][m]=jZq_em[ilinmom][ijack][mr];
                }
                
                Zq_err_r[ilinmom][m]=Zq_err[ilinmom][mr];
                Zq_em_err_r[ilinmom][m]=Zq_em_err[ilinmom][mr];
            }
            
            vvd_t jZq_pars_mom_r = polyfit(coord_q,2,Zq_err_r[ilinmom],jZq_r[ilinmom],x_min_q,x_max_q);
            vvd_t jZq_em_pars_mom_r = polyfit(coord_q,2,Zq_em_err_r[ilinmom],jZq_em_r[ilinmom],x_min_q,x_max_q);
            
            for(int ijack=0; ijack<njacks; ijack++)
            {
                (out.jZq)[ilinmom][ijack][r]=jZq_pars_mom_r[ijack][0];
                (out.jZq_em)[ilinmom][ijack][r]=jZq_em_pars_mom_r[ijack][0];
            }
        }
    }
    
    //extrapolate bilinears
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        for(int r1=0; r1<_nr; r1++)
            for(int r2=0; r2<_nr; r2++)
            {
                vvd_t coord_bil(vd_t(0.0,_nm*(_nm+1)/2),3); // coords at fixed r1 and r2
                
                vvvvd_t jG_0_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),_bilmoms);
                vvvvd_t jG_em_r1_r2(vvvd_t(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),njacks),nbil),_bilmoms);
                
                vvvd_t G_0_err_r1_r2(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),_bilmoms);
                vvvd_t G_em_err_r1_r2(vvd_t(vd_t(0.0,_nm*(_nm+1)/2),nbil),_bilmoms);

                int ieq=0;
                for(int m1=0; m1<_nm; m1++)
                    for(int m2=m1; m2<_nm; m2++)
                    {
                        int mr1 = r1 + _nr*m1;
                        int mr2 = r2 + _nr*m2;

                        coord_bil[0][ieq] = 1.0;
                        if(UseEffMass==0)
                        {
                            coord_bil[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
                            coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];    // 1/(am1+am2)
                        }
                        else if(UseEffMass==1)
                        {
                            coord_bil[1][ieq] = pow((M_eff[mr1][mr2]+M_eff[mr2][mr1])/2.0,2.0);   //M^2 (averaged over equivalent combinations)
                            coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];  //1/M^2
                        }
                    
                        for(int ibil=0;ibil<nbil;ibil++)
                        {
                            for(int ijack=0;ijack<njacks;ijack++)
                            {
                                jG_0_r1_r2[ibilmom][ibil][ijack][ieq] = (jG_0[ibilmom][ibil][ijack][mr1][mr2]/*+jG_0[ibilmom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0*/);
                                jG_em_r1_r2[ibilmom][ibil][ijack][ieq] = (jG_em[ibilmom][ibil][ijack][mr1][mr2]/*+jG_em[ibilmom][ibil][ijack][r1+_nr*m2][r2+_nr*m1])/2.0*/);
                            }
                            
                            G_0_err_r1_r2[ibilmom][ibil][ieq] = (G_0_err[ibilmom][ibil][mr1][mr2]/* + G_0_err[ibilmom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0*/);
                            G_em_err_r1_r2[ibilmom][ibil][ieq] = (G_em_err[ibilmom][ibil][mr1][mr2] /*+ G_em_err[ibilmom][ibil][r1+_nr*m2][r2+_nr*m1])/2.0*/);
                        }
                        
                        ieq++;
                    }
                
                for(int ibil=0;ibil<nbil;ibil++)
                {
                    vvd_t jG_0_pars_mom_ibil_r1_r2 = polyfit(coord_bil,npar[ibil],G_0_err_r1_r2[ibilmom][ibil],jG_0_r1_r2[ibilmom][ibil],x_min,x_max);
                    vvd_t jG_em_pars_mom_ibil_r1_r2 = polyfit(coord_bil,npar[ibil],G_em_err_r1_r2[ibilmom][ibil],jG_em_r1_r2[ibilmom][ibil],x_min,x_max);
                    
                    for(int ijack=0;ijack<njacks;ijack++)
                    {
//                        if(ibil==0 or ibil==2)
//                            for(int ieq=0;ieq<neq;ieq++)
//                            {
//                                // Goldstone pole subtraction from bilinears
//                                jG_0_ave_r[imom][ibil][ijack][ieq] -= jG_0_pars_mom[ibil][ijack][2];
//                                jG_em_ave_r[imom][ibil][ijack][ieq] -= jG_em_pars_mom[ibil][ijack][2];
//                            }
                        
                        // extrapolated value
                        (out.jG_0)[ibilmom][ibil][ijack][r1][r2] = jG_0_pars_mom_ibil_r1_r2[ijack][0];
                        (out.jG_em)[ibilmom][ibil][ijack][r1][r2] = jG_em_pars_mom_ibil_r1_r2[ijack][0];
                    }
                }
            }
    }
    
    out.compute_Zbil();
    
    return out;
}

oper_t oper_t::subtract()
{
    cout<<"Subtracting the O(a2) effects"<<endl<<endl;
    
    oper_t out=(*this);
    
//    resize_output(out);
    out.allocate();
    
#pragma omp parallel for collapse(3)
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0; mr1<_nmr; mr1++)
            {
                (out.jZq)[ilinmom][ijack][mr1] = jZq[ilinmom][ijack][mr1] - subtraction_q(ilinmom,LO);
                (out.jZq_em)[ilinmom][ijack][mr1] = jZq_em[ilinmom][ijack][mr1] + /*(!)*/ subtraction_q(ilinmom,EM)*jZq[ilinmom][ijack][mr1];
                // N.B.: the subtraction gets an extra minus sign due to the definition of the e.m. expansion!
            }
    
#pragma omp parallel for collapse(5)
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        for(int ibil=0;ibil<5;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr1=0; mr1<_nmr; mr1++)
                    for(int mr2=0; mr2<_nmr; mr2++)
                    {
                        (out.jG_0)[ibilmom][ibil][ijack][mr1][mr2] = jG_0[ibilmom][ibil][ijack][mr1][mr2] - subtraction(ibilmom,ibil,LO);
                        (out.jG_em)[ibilmom][ibil][ijack][mr1][mr2] = jG_em[ibilmom][ibil][ijack][mr1][mr2] - subtraction(ibilmom,ibil,EM)*jG_0[ibilmom][ibil][ijack][mr1][mr2];
                    }
    
    out.compute_Zbil();
    
    return out;
}


oper_t chiral_sea_extr(voper_t in)
{
    cout<<"Chiral sea extrapolation"<<endl<<endl;
    
    oper_t out=in[0];  //?
    
    int nmSea = in[0]._nm_Sea;
    int _linmoms = in[0]._linmoms;
    int _bilmoms = in[0]._bilmoms;
    
    //    resize_output(out);
    out.allocate();
    
    out.path_to_ens = in[0].path_to_beta;

    vd_t x(0.0,nmSea);
    
    vvd_t dy_Zq(vd_t(0.0,nmSea),_linmoms);
    vvd_t dy_Zq_em(vd_t(0.0,nmSea),_linmoms);
    
    vvvd_t dy_G(vvd_t(vd_t(0.0,nmSea),nbil),_bilmoms);
    vvvd_t dy_G_em(vvd_t(vd_t(0.0,nmSea),nbil),_bilmoms);

    vvvd_t y_Zq(vvd_t(vd_t(0.0,nmSea),njacks),_linmoms);
    vvvd_t y_Zq_em(vvd_t(vd_t(0.0,nmSea),njacks),_linmoms);
    
    vvvvd_t y_G(vvvd_t(vvd_t(vd_t(0.0,nmSea),njacks),nbil),_bilmoms);
    vvvvd_t y_G_em(vvvd_t(vvd_t(vd_t(0.0,nmSea),njacks),nbil),_bilmoms);
    
    // range for fit
    int x_min=0;
    int x_max=nmSea-1;
    
    for(int msea=0; msea<nmSea; msea++)
    {
        x[msea] = ( get<0>(ave_err(in[msea].eff_mass_sea)) )[0][0];

        for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
        {
            dy_Zq[ilinmom][msea] = (get<1>(ave_err(in[msea].jZq)))[ilinmom][0];
            dy_Zq_em[ilinmom][msea] = (get<1>(ave_err(in[msea].jZq_em)))[ilinmom][0];
            
            for(int ijack=0;ijack<njacks;ijack++)
            {
                y_Zq[ilinmom][ijack][msea] = in[msea].jZq[ilinmom][ijack][0];
                y_Zq_em[ilinmom][ijack][msea] = in[msea].jZq_em[ilinmom][ijack][0];
            }
        }
        
        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
            for(int ibil=0;ibil<nbil;ibil++)
            {
                dy_G[ibilmom][ibil][msea] = (get<1>(ave_err(in[msea].jG_0)))[ibilmom][ibil][0][0];
                dy_G_em[ibilmom][ibil][msea] = (get<1>(ave_err(in[msea].jG_em)))[ibilmom][ibil][0][0];
            
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    y_G[ibilmom][ibil][ijack][msea] = in[msea].jG_0[ibilmom][ibil][ijack][0][0];
                    y_G_em[ibilmom][ibil][ijack][msea] = in[msea].jG_em[ibilmom][ibil][ijack][0][0];
                }
            }
    }
    
    // extrapolate Zq
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
    {
        vvd_t coord_q(vd_t(0.0,nmSea),2); // coords at fixed r

        for(int msea=0; msea<nmSea; msea++)
        {
            coord_q[0][msea] = 1.0;
            if(UseEffMass==0)
            {
                cout<<" Impossible to extrapolate without using the effective mass. "<<endl;
                exit(0);
                //      coord_q[1][m]= mass_val[m];
            }
            else if(UseEffMass==1)
                coord_q[1][msea] = pow(x[msea],2.0);

            vvd_t jZq_pars_mom = polyfit(coord_q,2,dy_Zq[ilinmom],y_Zq[ilinmom],x_min,x_max);
            vvd_t jZq_em_pars_mom = polyfit(coord_q,2,dy_Zq_em[ilinmom],y_Zq_em[ilinmom],x_min,x_max);
            
            for(int ijack=0; ijack<njacks; ijack++)
            {
                (out.jZq)[ilinmom][ijack][0]=jZq_pars_mom[ijack][0];
                (out.jZq_em)[ilinmom][ijack][0]=jZq_em_pars_mom[ijack][0];
            }
        }
    }
    
    // extrapolate bilinears
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        vvd_t coord_bil(vd_t(0.0,nmSea),2); // linear fit in sea extrapolation
        
        for(int msea=0; msea<nmSea; msea++)
        {
            coord_bil[0][msea] = 1.0;
            if(UseEffMass==0)
            {
                cout<<" Impossible to extrapolate without using the effective mass. "<<endl;
                exit(0);
//                coord_bil[1][ieq] = mass_val[m1]+mass_val[m2];  // (am1+am2)
//                coord_bil[2][ieq] = 1.0/coord_bil[1][ieq];    // 1/(am1+am2)
            }
            else if(UseEffMass==1)
                coord_bil[1][msea] = pow(x[msea],2.0);
            
            
            for(int ibil=0;ibil<nbil;ibil++)
            {
                vvd_t jG_0_pars_mom_ibil = polyfit(coord_bil,2,dy_G[ibilmom][ibil],y_G[ibilmom][ibil],x_min,x_max);
                vvd_t jG_em_pars_mom_ibil = polyfit(coord_bil,2,dy_G_em[ibilmom][ibil],y_G_em[ibilmom][ibil],x_min,x_max);
                
                for(int ijack=0;ijack<njacks;ijack++)
                {
                    // extrapolated value
                    (out.jG_0)[ibilmom][ibil][ijack][0][0] = jG_0_pars_mom_ibil[ijack][0];
                    (out.jG_em)[ibilmom][ibil][ijack][0][0] = jG_em_pars_mom_ibil[ijack][0];
                }
            }
        }
    }
    
    out.compute_Zbil();
    
    return out;
}

oper_t theta_average( voper_t in)
{
    cout<<"Theta average"<<endl<<endl;
    
    oper_t out=in[0];  //?
    
    int _linmoms = in[0]._linmoms;
    int _bilmoms = in[0]._bilmoms;
    
    out.allocate();
    out.path_to_ens = in[0].path_to_beta;
    
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            out.jZq[ilinmom][ijack][0] = 0.5*(in[0].jZq[ilinmom][ijack][0] + in[1].jZq[ilinmom][ijack][0]);
            out.jZq_em[ilinmom][ijack][0] = 0.5*(in[0].jZq_em[ilinmom][ijack][0] + in[1].jZq_em[ilinmom][ijack][0]);
        }
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        for(int ibil=0;ibil<nbil;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
            {
                out.jZ[ibilmom][ibil][ijack][0][0] = 0.5*(in[0].jZ[ibilmom][ibil][ijack][0][0] + in[1].jZ[ibilmom][ibil][ijack][0][0]);
                out.jZ_em[ibilmom][ibil][ijack][0][0] = 0.5*(in[0].jZ[ibilmom][ibil][ijack][0][0] + in[1].jZ[ibilmom][ibil][ijack][0][0]);
            }
    
    return out;
}


oper_t oper_t::evolve(const int b)
{
    cout<<"Evolving the Z's to the scale 1/a"<<endl<<endl;
    
    oper_t out=(*this);

    double cq=0.0;
    vd_t cO(0.0,5);
    
    double _ainv=ainv[b];
    
    for(int ilinmom=0;ilinmom<_linmoms;ilinmom++)
    {
        cq=q_evolution_to_RIp_ainv(Nf,_ainv,p2[ilinmom]);
        
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0; mr1<_nmr; mr1++)
            {
                (out.jZq)[ilinmom][ijack][mr1] = jZq[ilinmom][ijack][mr1]/cq;
                (out.jZq_em)[ilinmom][ijack][mr1] = jZq_em[ilinmom][ijack][mr1]/cq;
            }
    }

    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        // Note that ZV  ZA are RGI because they're protected by the WIs
        cO[0]=S_evolution_to_RIp_ainv(Nf,_ainv,p2[ibilmom]); //S
        cO[1]=1.0;                                       //A
        cO[2]=P_evolution_to_RIp_ainv(Nf,_ainv,p2[ibilmom]); //P
        cO[3]=1.0;                                       //V
        cO[4]=T_evolution_to_RIp_ainv(Nf,_ainv,p2[ibilmom]); //T
        
        for(int ibil=0;ibil<5;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr1=0; mr1<_nmr; mr1++)
                    for(int mr2=0; mr2<_nmr; mr2++)
                    {
                        (out.jZ)[ibilmom][ibil][ijack][mr1][mr2] = jZ[ibilmom][ibil][ijack][mr1][mr2]/cO[ibil];
                        (out.jZ_em)[ibilmom][ibil][ijack][mr1][mr2] = jZ_em[ibilmom][ibil][ijack][mr1][mr2]/cO[ibil];
                    }
        
    }
    
    return out;
}

int mom_list_xyz(vector<coords_t> &mom_list, const size_t imom)
{
    return abs(mom_list[imom][1])*abs(mom_list[imom][2])*abs(mom_list[imom][3]);
}

oper_t oper_t::average_equiv_moms()
{
    cout<<"Averaging over the equivalent momenta -- ";
    
    oper_t out=(*this);
    
    // Find equivalent linmoms
    int tag=0, tag_aux=0;
    double eps=1.0e-15;
    
    vector<int> tag_lin_vector;
    tag_lin_vector.push_back(0);
    
    // Tag assignment to linmoms
    for(int imom=0;imom<_linmoms;imom++)
    {
        int count_no=0;
        
        for(int j=0;j<imom;j++)
        {
            if( 2.0*abs(p2_tilde[j]-p2_tilde[imom])<eps*(p2_tilde[j]+p2_tilde[imom]) && mom_list_xyz(mom_list,j)==mom_list_xyz(mom_list,imom) &&
               2.0*abs(abs(p[j][0])-abs(p[imom][0]))<eps*(abs(p[j][0])+abs(p[imom][0])) )
            {
                tag_aux = tag_lin_vector[j];
            }else count_no++;
            
            if(count_no==imom)
            {
                tag++;
                tag_lin_vector.push_back(tag);
            }else if(j==imom-1)
            {
                tag_lin_vector.push_back(tag_aux);
            }
        }
    }
    
    // number of equivalent linmoms
    int neq_lin_moms = tag+1;
    
//    int neqmoms = neq_lin_moms;
    
    out._linmoms=neq_lin_moms;
    cout<<"found: "<<out._linmoms<<" equivalent linmoms ";
    (out.linmoms).resize(out._linmoms);
    
    vector<double> p2_tilde_eqmoms(out._linmoms,0.0);


    // count the different tags
    vector<int> count_tag_lin_vector(out._linmoms);
    int count=0;
    for(int tag=0;tag<out._linmoms;tag++)
    {
        count=0;
        for(int imom=0;imom<_linmoms;imom++)
        {
            if(tag_lin_vector[imom]==tag) count++;
        }
        count_tag_lin_vector[tag]=count;
    }
    
    for(int tag=0;tag<out._linmoms;tag++)
        for(int imom=0;imom<_linmoms;imom++)
        {
            if(tag_lin_vector[imom]==tag)
            {
                // fill the new linmoms and p2tilde
                out.linmoms[tag] = {imom};
                p2_tilde_eqmoms[tag] = p2_tilde[imom];
//                cout<<"{"<<tag<<"}"<<endl;
            }
        }
    
//    PRINT(p2_tilde_eqmoms);
    print_vec(p2_tilde_eqmoms,path_print+"p2_tilde_eqmoms.txt");

    
    // Find equivalent bilmoms
    tag=0, tag_aux=0;
    
    vector<int> tag_bil_vector;
    tag_bil_vector.push_back(0);
    
    
    //Tag assignment to bilmoms
    for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
    {
        int count_no=0;
        
        const int imom1=bilmoms[ibilmom][1]; // p1
        const int imom2=bilmoms[ibilmom][2]; // p2
        
        for(int j=0;j<ibilmom;j++)
        {
            const int imomA=bilmoms[j][1]; // p1
            const int imomB=bilmoms[j][2]; // p2
            
            if( (tag_lin_vector[imom1]==tag_lin_vector[imomA] and tag_lin_vector[imom2]==tag_lin_vector[imomB])
               or (tag_lin_vector[imom1]==tag_lin_vector[imomB] and tag_lin_vector[imom2]==tag_lin_vector[imomA]))
//            if(tag_lin_vector[imom1]+tag_lin_vector[imom2]==tag_lin_vector[imomA]+tag_lin_vector[imomB] and
//               tag_lin_vector[imom1]*tag_lin_vector[imom2]==tag_lin_vector[imomA]*tag_lin_vector[imomB])
            {
                tag_aux=tag_bil_vector[j];
            }else count_no++;
            
            if(count_no==ibilmom)
            {
                tag++;
                tag_bil_vector.push_back(tag);
            }else if(j==ibilmom-1)
            {
                tag_bil_vector.push_back(tag_aux);
            }
        }
    }
    
    // number of equivalent bilmoms
    int neq_bil_moms = tag+1;
    
    out._bilmoms=neq_bil_moms;
    cout<<"and "<<neq_bil_moms<<" equivalent bilmoms "<<endl<<endl;
    (out.bilmoms).resize(out._bilmoms);
    
    // count the different tags
    vector<int> count_tag_bil_vector(out._bilmoms);
    count=0;
    for(int tag=0;tag<out._bilmoms;tag++)
    {
        count=0;
        for(int imom=0;imom<_bilmoms;imom++)
        {
            if(tag_bil_vector[imom]==tag) count++;
        }
        count_tag_bil_vector[tag]=count;
    }
    
    for(int tag=0;tag<out._bilmoms;tag++)
        for(int ibilmom=0;ibilmom<_bilmoms;ibilmom++)
        {
            if(tag_bil_vector[ibilmom]==tag)
            {
                // fill the new bilmoms
                const int imom0=bilmoms[ibilmom][0]; // k
                const int imom1=bilmoms[ibilmom][1]; // p1
                const int imom2=bilmoms[ibilmom][2]; // p2
                
                out.bilmoms[tag] = {imom0,imom1,imom2};
//                cout<<tag<<" {"<<imom0<<","<<imom1<<","<<imom2<<"}"<<endl;
            }
        }
    
//    resize_output(out);
    out.allocate();
    
    // initialize to zero
#pragma omp parallel for collapse(3)
    for(int tag=0;tag<neq_lin_moms;tag++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr1=0; mr1<_nmr; mr1++)
            {
                (out.jZq)[tag][ijack][mr1]=0.0;
                (out.jZq_em)[tag][ijack][mr1]=0.0;
            }
#pragma omp parallel for collapse(5)
    for(int tag=0;tag<neq_bil_moms;tag++)
        for(int ibil=0;ibil<5;ibil++)
            for(int ijack=0;ijack<njacks;ijack++)
                for(int mr1=0; mr1<_nmr; mr1++)
                    for(int mr2=0; mr2<_nmr; mr2++)
                    {
                        (out.jZ)[tag][ibil][ijack][mr1][mr2]=0.0;
                        (out.jZ_em)[tag][ibil][ijack][mr1][mr2]=0.0;
                    }
    
    // average over the equivalent momenta
    for(int tag=0;tag<neq_lin_moms;tag++)
        for(int imom=0;imom<_linmoms;imom++)
        {
            if(tag_lin_vector[imom]==tag)
            {
                for(int ijack=0;ijack<njacks;ijack++)
                    for(int mr1=0; mr1<_nmr; mr1++)
                    {
                        (out.jZq)[tag][ijack][mr1]+=jZq[imom][ijack][mr1]/count_tag_lin_vector[tag];
                        (out.jZq_em)[tag][ijack][mr1]+=jZq_em[imom][ijack][mr1]/count_tag_lin_vector[tag];
                    }
            }
        }
    for(int tag=0;tag<neq_bil_moms;tag++)
        for(int imom=0;imom<_bilmoms;imom++)
        {
            if(tag_bil_vector[imom]==tag)
            {
                for(int ibil=0;ibil<5;ibil++)
                    for(int ijack=0;ijack<njacks;ijack++)
                        for(int mr1=0; mr1<_nmr; mr1++)
                            for(int mr2=0; mr2<_nmr; mr2++)
                            {
                                (out.jZ)[tag][ibil][ijack][mr1][mr2]+=jZ[imom][ibil][ijack][mr1][mr2]/count_tag_bil_vector[tag];
                                (out.jZ_em)[tag][ibil][ijack][mr1][mr2]+=jZ_em[imom][ibil][ijack][mr1][mr2]/count_tag_bil_vector[tag];
                            }
            }
        }
    
    return out;
}

voper_t a2p2_extr(voper_t in /*, const int LO_or_EM*/)  // M1 method
{
    voper_t out;
    
    //    int neq_moms = (out.jZq).size();
    int _linmoms=in[0]._linmoms;
    int _bilmoms=in[0]._bilmoms;
    
    vector<double> p2_tilde_eqmoms(_linmoms);
    read_vec(p2_tilde_eqmoms,in[0].path_print+"p2_tilde_eqmoms.txt");
    
    double b0 = in[1]._beta; // b0 is the 'central' value of beta.
    vd_t b(nbeta);
    for(auto &i : b)
    {
        b[i] = in[i]._beta;
    }
    
    for(int ibeta=0; ibeta<nbeta; ibeta++)
    {
        out[ibeta]=in[ibeta];
        
        out[ibeta]._linmoms=1;
        out[ibeta]._bilmoms=1;

        out[ibeta].allocate();
        
        for(int LO_or_EM=0; LO_or_EM<2; LO_or_EM++)
        {
            vvd_t jZq_out(vd_t(0.0,_linmoms),njacks);
            vvvd_t jZ_out(vvd_t(vd_t(0.0,_bilmoms),njacks),nbil);
            
            vd_t Zq_err(0.0,_linmoms);
            vvd_t Z_err(vd_t(0.0,_bilmoms),nbil);
            
            if(LO_or_EM==0)
            {
                cout<<"-- Leading Order --"<<endl;
                
#pragma omp parallel for collapse(2)
                for(int imom=0; imom<_linmoms; imom++)
                    for(int ijack=0; ijack<njacks; ijack++)
                        jZq_out[ijack][imom] = in[ibeta].jZq[imom][ijack][0];
                
#pragma omp parallel for collapse(3)
                for(int imom=0; imom<_bilmoms; imom++)
                    for(int ijack=0; ijack<njacks; ijack++)
                        for(int ibil=0; ibil<nbil; ibil++)
                            jZ_out[ibil][ijack][imom] = in[ibeta].jZ[imom][ibil][ijack][0][0];
                
                vvd_t Zq_err_tmp = get<1>(ave_err(in[ibeta].jZq));
                vvvvd_t Z_err_tmp = get<1>(ave_err(in[ibeta].jZ));
                
                for(int imom=0; imom<_linmoms; imom++)
                    Zq_err[imom] = Zq_err_tmp[imom][0];
                
                for(int imom=0; imom<_bilmoms; imom++)
                    for(int ibil=0; ibil<nbil; ibil++)
                        Z_err[ibil][imom] = Z_err_tmp[imom][ibil][0][0];
                
            }
            else if(LO_or_EM==1)
            {
                cout<<"-- EM Correction --"<<endl;
                
#pragma omp parallel for collapse(2)
                for(int imom=0; imom<_linmoms; imom++)
                    for(int ijack=0; ijack<njacks; ijack++)
                        jZq_out[ijack][imom] = in[ibeta].jZq_em[imom][ijack][0];
                
#pragma omp parallel for collapse(3)
                for(int imom=0; imom<_bilmoms; imom++)
                    for(int ijack=0; ijack<njacks; ijack++)
                        for(int ibil=0; ibil<nbil; ibil++)
                            jZ_out[ibil][ijack][imom] = in[ibeta].jZ_em[imom][ibil][ijack][0][0];
                
                vvd_t Zq_err_tmp = get<1>(ave_err(in[ibeta].jZq_em));
                vvvvd_t Z_err_tmp = get<1>(ave_err(in[ibeta].jZ_em));
                
                for(int imom=0; imom<_linmoms; imom++)
                    Zq_err[imom] = Zq_err_tmp[imom][0];
                
                for(int imom=0; imom<_bilmoms; imom++)
                    for(int ibil=0; ibil<nbil; ibil++)
                        Z_err[ibil][imom] = Z_err_tmp[imom][ibil][0][0];
            }
            
            // Simultaneous extrapolation: y = A + B(g2)*a2p2
            //  B(g^2) = B(g0^2) + C*(g^2 - g0^2)  where g0 is the 'central' value of the coupling constant
            //         = B(g0^2) + C*6*(b0 - b)/(b0*b)
            int npar = 3;
            
            //linear fit Zq
            int range_min=0;  //a2p2~1
            int range_max=_linmoms;
            double p_min_value=p2min;
            
            vvd_t coord_lin_linear(vd_t(0.0,_linmoms),npar);
            
            for(int i=0; i<range_max; i++)
            {
                coord_lin_linear[0][i] = 1.0;  //costante
                coord_lin_linear[1][i] = p2_tilde_eqmoms[i];   //p^2
                coord_lin_linear[2][i] = p2_tilde_eqmoms[i]*6.0*(b0-b[ibeta])/(b0*b[ibeta]);
            }
            
            vd_t jZq_out_par_ijack(0.0,npar);
            
            double Zq_ave_cont=0.0, sqr_Zq_ave_cont=0.0, Zq_err_cont=0.0;
            
            for(int ijack=0; ijack<njacks; ijack++)
            {
                jZq_out_par_ijack=fit_continuum(coord_lin_linear,Zq_err,jZq_out[ijack],range_min,range_max,p_min_value);
                
                Zq_ave_cont += jZq_out_par_ijack[0]/njacks;
                sqr_Zq_ave_cont += jZq_out_par_ijack[0]*jZq_out_par_ijack[0]/njacks;
                
                if(LO_or_EM==0)
                {
                    (out[ibeta].jZq)[0][ijack][0] = jZq_out_par_ijack[0];
                }
                else if(LO_or_EM==1)
                {
                    (out[ibeta].jZq_em)[0][ijack][0] = jZq_out_par_ijack[0];
                }
            }
            
            Zq_err_cont=sqrt((double)(njacks-1))*sqrt(sqr_Zq_ave_cont-Zq_ave_cont*Zq_ave_cont);
            
            cout<<"ZQ = "<<Zq_ave_cont<<" +/- "<<Zq_err_cont<<endl;
            
            //linear fit Z
            range_min=0;  //a2p2~1
            range_max=_bilmoms;
            
            vvd_t coord_bil_linear(vd_t(0.0,_bilmoms),npar);
            
            for(int i=0; i<range_max; i++)
            {
                //        int imomk = (out.bilmoms)[i][0];
                int imomk = i;      /// it will work temporarily only for RIMOM (!!!!!!!)
                
                coord_bil_linear[0][i] = 1.0;  //costante
                coord_bil_linear[1][i] = p2_tilde_eqmoms[imomk];   //p^2
                coord_bil_linear[2][i] = p2_tilde_eqmoms[imomk]*6.0*(b0-b[ibeta])/(b0*b[ibeta]);
            }
            
            vvd_t jZ_out_par_ijack(vd_t(0.0,npar),nbil);
            vd_t Z_ave_cont(0.0,nbil), sqr_Z_ave_cont(0.0,nbil), Z_err_cont(0.0,nbil);
            
            for(int ijack=0; ijack<njacks; ijack++)
                for(int ibil=0; ibil<nbil; ibil++)
                {
                    jZ_out_par_ijack[ibil]=fit_continuum(coord_bil_linear,Z_err[ibil],jZ_out[ibil][ijack],range_min,range_max,p_min_value);
                    
                    Z_ave_cont[ibil] += jZ_out_par_ijack[ibil][0]/njacks;
                    sqr_Z_ave_cont[ibil] += jZ_out_par_ijack[ibil][0]*jZ_out_par_ijack[ibil][0]/njacks;
                    
                    if(LO_or_EM==0)
                    {
                        (out[ibeta].jZ)[0][ibil][ijack][0][0] = jZ_out_par_ijack[ibil][0];
                    }
                    else if(LO_or_EM==1)
                    {
                        (out[ibeta].jZ_em)[0][ibil][ijack][0][0] = jZ_out_par_ijack[ibil][0];
                    }
                }
            
            for(int ibil=0; ibil<nbil;ibil++)
                Z_err_cont[ibil]=sqrt((double)(njacks-1))*sqrt(fabs(sqr_Z_ave_cont[ibil]-Z_ave_cont[ibil]*Z_ave_cont[ibil]));
            
            vector<string> bil={"S","A","P","V","T"};
            
            for(int ibil=0; ibil<nbil;ibil++)
            {
                cout<<"Z"<<bil[ibil]<<" = "<<Z_ave_cont[ibil]<<" +/- "<<Z_err_cont[ibil]<<endl;
            }
            
            //    vector<double> pert={-0.0695545,-0.100031,-0.118281,-0.130564,-0.108664}; // Martinelli-Zhang
            //
            //    if(LO_or_EM==1)
            //    {
            //        cout<<"Z divided by the perturbative estimates (to be evolved in MSbar"
            //    for(int ibil=0;i<nbil;ibil++)
            //    {
            //        cout<<"Z"<<bil[ibil]<<"(fact) = "<<A_bil[ibil]/pert[ibil]<<" +/- "<<A_err[ibil]/pert[ibil]<<endl;
            //    }
            //    }
            
            cout<<endl;
            
            
            
           
            
        } // close LO_or_EM loop
    }// close beta loop
    return out;
}

void oper_t::plot(const string suffix)
{
    oper_t in=(*this);
    
    Zq_tup Zq_ave_err = ave_err(in.jZq);
    Zq_tup Zq_em_ave_err = ave_err(in.jZq_em);
    
    Zbil_tup Zbil_ave_err = ave_err(in.jZ);
    Zbil_tup Zbil_em_ave_err = ave_err(in.jZ_em);
   
    vvd_t Zq_ave = get<0>(Zq_ave_err);        //[imom][mr]
    vvd_t Zq_em_ave = get<0>(Zq_em_ave_err);
    
    vvd_t Zq_err = get<1>(Zq_ave_err);        //[imom][mr]
    vvd_t Zq_em_err = get<1>(Zq_em_ave_err);
    
    vvvvd_t Z_ave = get<0>(Zbil_ave_err);    //[imom][ibil][mr1][mr2]
    vvvvd_t Z_em_ave = get<0>(Zbil_em_ave_err);
    
    vvvvd_t Z_err = get<1>(Zbil_ave_err);    //[imom][ibil][mr1][mr2]
    vvvvd_t Z_em_err = get<1>(Zbil_em_ave_err);
    
    vector<string> bil={"S","A","P","V","T"};
    
    ofstream Zq_data, Zq_em_data;
    vector<ofstream> Zbil_data(nbil), Zbil_em_data(nbil);
    
    Zq_data.open(path_to_ens+"plots/Zq"+(suffix!=""?("_"+suffix):string(""))+".txt");
    Zq_em_data.open(path_to_ens+"plots/Zq_EM"+(suffix!=""?("_"+suffix):string(""))+".txt");
    
    vector<double> p2t;
    
    if(in._linmoms==moms)
    {
//        cout<<"A"<<endl;
        p2t.resize(in._linmoms);
        read_vec(p2t,path_print+"p2_tilde.txt");
        
//        READ2(p2t,p2_tilde)
    }
    else
    {
//        cout<<"B"<<endl;
        p2t.resize(in._linmoms);
        read_vec(p2t,path_print+"p2_tilde_eqmoms.txt");

//        READ2(p2t,p2_tilde_eqmoms);
    }
    
    for(int imom=0; imom<in._linmoms; imom++)
    {
        Zq_data<<p2t[imom]<<"\t"<<Zq_ave[imom][0]<<"\t"<<Zq_err[imom][0]<<endl;
        Zq_em_data<<p2t[imom]<<"\t"<<Zq_em_ave[imom][0]<<"\t"<<Zq_em_err[imom][0]<<endl;
    }
    
    for(int ibil=0;ibil<nbil;ibil++)
    {
        Zbil_data[ibil].open(path_to_ens+"plots/Z"+bil[ibil]+(suffix!=""?("_"+suffix):string(""))+".txt");
        Zbil_em_data[ibil].open(path_to_ens+"plots/Z"+bil[ibil]+"_EM"+(suffix!=""?("_"+suffix):string(""))+".txt");
        
        for(int imom=0; imom<in._bilmoms; imom++)
        {
//            int imomq = in.bilmoms[imom][0];
//            cout<<"imomq: "<<imomq<<endl;
//            int imomk = in.linmoms[imomq][0];
            int imomk = imom;   // NB: it works only for RIMOM!
            
            Zbil_data[ibil]<<p2t[imomk]<<"\t"<<Z_ave[imom][ibil][0][0]<<"\t"<<Z_err[imom][ibil][0][0]<<endl;
            Zbil_em_data[ibil]<<p2t[imomk]<<"\t"<<Z_em_ave[imom][ibil][0][0]<<"\t"<<Z_em_err[imom][ibil][0][0]<<endl;
        }
    }
}
