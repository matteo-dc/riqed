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
    if(ratio_analysis)
        nloop=2;
    
    vvoper_t oper_for_ratio(voper_t(nloop),nbeta);
    voper_t ratio(nbeta);
    recompute_basic = false;
        
    for(int loop=0;loop<nloop;loop++)
    {
        vvoper_t rave_ave_chir_sea(voper_t(ntheta),nbeta);
        voper_t  rave_ave_chir_sea_theta(nbeta);
        voper_t  rave_ave_chir_sea_theta_extr(nbeta);
        
        if(loop>0) recompute_basic=true;

        //////////
        
        for(int b=0; b<nbeta; b++)
        {
            vvoper_t basic(voper_t(nm_Sea[b]),ntheta);
            vvoper_t ave(voper_t(nm_Sea[b]),ntheta);
            
            vvoper_t rave(voper_t(nm_Sea[b]),ntheta);
            
            vvoper_t val_chir(voper_t(nm_Sea[b]),ntheta);
            vvoper_t rave_ave_chir_M2(voper_t(nm_Sea[b]),ntheta);
            
            vvoper_t rave_ave_chir_sub(voper_t(nm_Sea[b]),ntheta);
            
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
                            val_chir[th][m] = rave[th][m];
                        else
                        {
                            val_chir[th][m] = rave[th][m].chiral_extr();
                            val_chir[th][m].plot("chir");
                        }
                        
//                        /* interpolation to p2ref (M2 method) */
//                        
//                        rave_ave_chir_M2[th][m] = rave_ave_chir[th][m].interpolate_to_p2ref(b);
//                        rave_ave_chir_M2[th][m].plot("rave_ave_chir_M2");
                    }
                    
                } //close nm_sea
                
                /*  sea chiral extr  */
                
                if(!only_basic)
                {
                    if(nm_Sea[b]>1 and (!free_analysis or !recompute_basic))
                    {
                        rave_ave_chir_sea[b][th] = chiral_sea_extr(val_chir[th]);
                        rave_ave_chir_sea[b][th].plot(theta_label[th]+"/rave_ave_chir_sea");
                    }
                    else
                        rave_ave_chir_sea[b][th] = val_chir[th][0];
                }
            } //close ntheta
            
            if(!only_basic)
            {
                if(ntheta>1)
                {
                    /*  theta average  */
                    
                    rave_ave_chir_sea_theta[b] = theta_average(rave_ave_chir_sea[b]);
                    rave_ave_chir_sea_theta[b].plot("rave_ave_chir_sea_theta");
                    
                    /*  evolution to 1/a  */
                    
//                    rave_ave_chir_sub_sea_theta_evo[b] = rave_ave_chir_sub_sea_theta[b].evolve(b);
//                    rave_ave_chir_sub_sea_theta_evo[b].plot("rave_ave_chir_sub_sea_theta_evo");
                }
                else
                    rave_ave_chir_sea_theta[b] = rave_ave_chir_sea[b][0];
            }
            
            /* store for ratio */
            
            if(ratio_analysis)
                oper_for_ratio[b][loop] = rave_ave_chir_sea_theta[b];
            
        } //close nbeta
        
    } //close nloop
    
    if(ratio_analysis)
        for(int b=0; b<nbeta; b++)
        {
            ratio[b] = compute_ratio(oper_for_ratio[b]);
            ratio[b].plot("ratio");
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
