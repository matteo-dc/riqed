#ifdef HAVE_CONFIG_H
#include <config.hpp>
#endif

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "aliases.hpp"
#include "read.hpp"
#include "read_input.hpp"
#include "Dirac.hpp"
#include "global.hpp"
#include "operations.hpp"
#include "print.hpp"
#include "fit.hpp"

int main(int narg,char **arg)
{
    omp_set_nested(1);
#pragma omp parallel
#pragma omp master
    {
        system("clear");
        cout<<"Using "<<omp_get_num_threads()<<" threads"<<endl<<endl;
    }
    
    char path_glb[128]="input_glb.txt";
    
    if (narg>2){cerr<<"Number of arguments not valid."<<endl;
                exit(0);}
    if (narg==2){
        string path=arg[1];
        strcpy(path_glb,path.c_str());
    }
    
    cout<<"Reading global input from \""<<path_glb<<"\"."<<endl;
    read_input_glb(path_glb);
    
    int nloop;
    if(free_analysis or inte_analysis)
        nloop=1;
    if(eta_analysis)
        nloop=2;
    
    vvvvoper_t oper_for_eta(vvvoper_t(vvoper_t(voper_t(nloop),nm_Sea_max),ntheta),nbeta);
    vvvoper_t eta(vvoper_t(voper_t(nm_Sea_max),ntheta),nbeta);
    vvvoper_t etaM1(vvoper_t(voper_t(nm_Sea_max),nbeta),ntheta);
    vvvoper_t etaM2(vvoper_t(voper_t(nm_Sea_max),nbeta),ntheta);
    vvoper_t etaM1_sea(voper_t(nbeta),ntheta);
    vvoper_t etaM2_sea(voper_t(nbeta),ntheta);
    
    recompute_basic = false;
        
    for(int loop=0;loop<nloop;loop++)
    {
        if(nloop>1)
            cout<<" ********** LOOP "<<loop<<" ********** "<<endl;
        
        vvoper_t sea_chir(voper_t(ntheta),nbeta);
        voper_t  theta_ave(nbeta);
        voper_t  p2extr(nbeta);
        
        if(loop>0) recompute_basic=true;

        //////////
        
        for(int b=0; b<nbeta; b++)
        {
            vvoper_t basic(voper_t(nm_Sea[b]),ntheta);
            vvoper_t ave(voper_t(nm_Sea[b]),ntheta);
            
            vvoper_t rave(voper_t(nm_Sea[b]),ntheta);
            
            vvoper_t val_chir(voper_t(nm_Sea[b]),ntheta);
//            vvoper_t interpol_M2(voper_t(nm_Sea[b]),ntheta);
            
            
            for(int th=0; th<ntheta; th++)
            {
                for(int m=0; m<nm_Sea[b]; m++)
                {
                    /*  basic  */
                    
                    basic[th][m].create_basic(b,th,m);
                    if(!load_ave) basic[th][m].plot("");
                    
                    /*  average over equivalent momenta  */
                    
                    ave[th][m] = basic[th][m].average_equiv_moms();
                    if(!load_ave) ave[th][m].plot("ave");
                    
                    /*  average r  */
                    
                    rave[th][m] = ave[th][m].average_r();
                    if(!load_ave) rave[th][m].plot("rave");
                    
                    /* store averaged ingredients */
                    if(!load_ave) rave[th][m].print("rave");
                    
                    if(!only_basic)
                    {
                        /* load averaged ingredients if needed */
                        if(load_ave) rave[th][m].load("rave");
                        
                        /*  valence chiral extr  */
                        if(free_analysis or recompute_basic)
                        {
                            val_chir[th][m] = rave[th][m];
                            val_chir[th][m].plot("chir");
                            
                            /* store extrapolated ingredients */
                            if(!load_chir) val_chir[th][m].printZ("chir");
                        }
                        else
                        {
                            val_chir[th][m] = rave[th][m].chiral_extr();
                            val_chir[th][m].plot("chir");
                            
                            /* store extrapolated ingredients */
                            if(!load_chir) val_chir[th][m].printZ("chir");
                        }
                        
                        
//                        /* interpolation to p2ref (M2 method) */
//                        
//                        interpol_M2[th][m] = val_chir[th][m].interpolate_to_p2ref(b);
//                        interpol_M2[th][m].plot("interpol_M2");
                   
                    } //close if(!only_basic)
                    

                    if(eta_analysis)
                        oper_for_eta[b][th][m][loop] = val_chir[th][m];

                    
                } //close nm_sea
                
//                /*  sea chiral extr  */
//                
//                if(!only_basic)
//                {
//                    if(nm_Sea[b]>1 and (!free_analysis or !recompute_basic))
//                    {
//                        sea_chir[b][th] = chiral_sea_extr(val_chir[th]);
//                        sea_chir[b][th].plot(theta_label[th]+"/sea");
//                    }
//                    else
//                        sea_chir[b][th] = val_chir[th][0];
//                }

            } //close ntheta
            
//            if(!only_basic)
//            {
//                if(ntheta>1)
//                {
//                    /*  theta average  */
//                    
//                    theta_ave[b] = theta_average(sea_chir[b]);
//                    theta_ave[b].plot("theta");
//                    
//                    /*  evolution to 1/a  */
//                    
////                    rave_ave_chir_sub_sea_theta_evo[b] = rave_ave_chir_sub_sea_theta[b].evolve(b);
////                    rave_ave_chir_sub_sea_theta_evo[b].plot("rave_ave_chir_sub_sea_theta_evo");
//                }
//                else
//                    theta_ave[b] = sea_chir[b][0];
//            }
            
            /* store for eta */
            
//            if(eta_analysis)
//                oper_for_eta[b][loop] = theta_ave[b];

         
        } //close nbeta
        
    } //close nloop
    
    if(eta_analysis)
    {
        
        for(int b=0; b<nbeta; b++)
        {
            for(int th=0; th<ntheta; th++)
            {
                eta[b][th].resize(nm_Sea[b]);
                etaM1[b][th].resize(nm_Sea[b]);
                etaM2[b][th].resize(nm_Sea[b]);
                
                for(int m=0; m<nm_Sea[b]; m++)
                {
                    eta[b][th][m] = compute_eta(oper_for_eta[b][th][m]);
                    eta[b][th][m].plot("eta");
                    
                    eta[b][th][m] = eta[b][th][m].evolve_mixed();
                    eta[b][th][m].plot("eta_evo");
                    
                    etaM2[th][b][m] = eta[b][th][m].interpolate_to_p2ref(b); /* (b) !?*/
                    etaM2[th][b][m].plot("eta_M2");
                    etaM1[th][b][m] = eta[b][th][m].a2p2_extr();
                    etaM1[th][b][m].plot("eta_M1");
                    
                } //close nm_sea
            } //close theta
        } //close beta
        
        for(int b=0; b<nbeta; b++)
        {
            for(int th=0; th<ntheta; th++)
            {
                etaM2_sea[th] = combined_chiral_sea_extr(etaM2[th]);
                etaM1_sea[th] = combined_chiral_sea_extr(etaM1[th]);
            }
            
//            etaM2_sea[th][b].plot("?");
//            etaM1_sea[th][b].plot("?");
        }

    }
    

//    if(nbeta>1 and only_basic==0)
//    {
//        /*  a2p2->0 extrapolation  */
//        
//        rave_ave_chir_sub_sea_theta_evo_extr = a2p2_extr(rave_ave_chir_sub_sea_theta_evo);
//        
//        /*  continuum limit  */
//        
//        cout<<"Continuum limit extrapolation:"<<endl<<endl;
////        a2p2_extr(rave_chir_sub_sea_theta_evo_ave,LO);
////        a2p2_extr(rave_chir_sub_sea_theta_evo_ave,EM);
//    }
    
    cout<<"DONE!"<<endl;
    
    return 0;
}
