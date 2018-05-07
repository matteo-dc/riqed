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

//debug
#include "subtraction.hpp"

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
    
    
    vvoper_t rave_ave_chir_sub_sea(voper_t(ntheta),nbeta);
    voper_t  rave_ave_chir_sub_sea_theta(nbeta);
    voper_t  rave_ave_chir_sub_sea_theta_evo(nbeta);
    voper_t  rave_ave_chir_sub_sea_theta_evo_extr(nbeta);
    
    for(int b=0; b<nbeta; b++)
    {
        vvvoper_t basic(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        vvvoper_t ave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
        vvvoper_t rave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        vvvoper_t rave_ave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
        vvvoper_t rave_ave_chir(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
        vvvoper_t rave_ave_chir_sub(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
        for(int th=0; th<ntheta; th++)
        {
            for(int m=0; m<nm_Sea[b]; m++)
            {
                /*  basic  */
                
                basic[b][th][m].create_basic(b,th,m);
                basic[b][th][m].plot("");
                ave[b][th][m] = basic[b][th][m].average_equiv_moms();
                ave[b][th][m].plot("ave");
                
                if(only_basic==0)
                {
                    /*  average r  */
                    
                    rave[b][th][m] = basic[b][th][m].average_r();
                    rave[b][th][m].plot("rave");
                    
                    /*  average over equivalent momenta  */
                    
                    rave_ave[b][th][m] = rave[b][th][m].average_equiv_moms();
                    rave_ave[b][th][m].plot("rave_ave");
                    
                    /*  valence chiral extr  */
                    
                    if(strcmp(analysis.c_str(),"free")!=0)
                        rave_ave_chir[b][th][m] = rave_ave[b][th][m].chiral_extr();
                    else
                        rave_ave_chir[b][th][m] = rave_ave[b][th][m];
                    rave_ave_chir[b][th][m].plot("rave_ave_chir");
                    
//                    /*  O(a2g2) subtraction  */
//                    
//                    rave_ave_chir_sub[b][th][m] = rave_ave_chir[b][th][m].subtract();
//                    rave_ave_chir_sub[b][th][m].plot("rave_ave_chir_sub");
                }
                
            } //close nm_sea
            
            if(nm_Sea[b]>1 and only_basic==0 and strcmp(analysis.c_str(),"free" )!=0)
            {
                /*  sea chiral extr  */
                
                rave_ave_chir_sub_sea[b][th] = chiral_sea_extr(rave_ave_chir_sub[b][th]);
                rave_ave_chir_sub_sea[b][th].plot(theta_label[th]+"/rave_ave_chir_sub_sea");
            }
            else if(only_basic==0)
            {
                rave_ave_chir_sub_sea[b][th] = rave_ave_chir_sub[b][th][0];
            }
        } //close ntheta
        
        if(ntheta>1 and only_basic==0)
        {
            /*  theta average  */
            
            rave_ave_chir_sub_sea_theta[b] = theta_average(rave_ave_chir_sub_sea[b]);
            rave_ave_chir_sub_sea_theta[b].plot("rave_ave_chir_sub_sea_theta");
            
//            /*  evolution to 1/a  */
//            
//            rave_ave_chir_sub_sea_theta_evo[b] = rave_ave_chir_sub_sea_theta[b].evolve(b);
//            rave_ave_chir_sub_sea_theta_evo[b].plot("rave_ave_chir_sub_sea_theta_evo");
        }
    } //close nbeta
    
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
