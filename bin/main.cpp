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

ORDER ord;
MESLEP_TYPES meslep_t;

int main()
{
    omp_set_nested(1);
#pragma omp parallel
#pragma omp master
    {
        system("clear");
        cout<<"Using "<<omp_get_num_threads()<<" threads"<<endl<<endl;
    }
    
    char path_glb[128]="input_glb.txt";
    cout<<"Reading global input"<<endl;
    read_input_glb(path_glb);
    
    
    vvoper_t rave_chir_sub_sea(voper_t(ntheta),nbeta);
    vvoper_t rave_chir_sub_sea_ave(voper_t(ntheta),nbeta);
    
    voper_t rave_chir_sub_sea_theta(nbeta);
    voper_t rave_chir_sub_sea_theta_ave(nbeta);
    
    voper_t rave_chir_sub_sea_theta_evo(nbeta);
    voper_t rave_chir_sub_sea_theta_evo_ave(nbeta);
    
    voper_t rave_chir_sub_sea_theta_evo_ave_extr(nbeta);
    
    for(int b=0; b<nbeta; b++)
    {
        vvvoper_t basic(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        vvvoper_t ave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
        vvvoper_t rave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        vvvoper_t rave_ave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
        vvvoper_t rave_chir(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        vvvoper_t rave_chir_ave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
        vvvoper_t rave_chir_sub(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        vvvoper_t rave_chir_sub_ave(vvoper_t(voper_t(nm_Sea[b]),ntheta),nbeta);
        
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
                    rave_ave[b][th][m] = rave[b][th][m].average_equiv_moms();
                    rave_ave[b][th][m].plot("rave_ave");
                    
                    /*  valence chiral extr  */
                    
                    rave_chir[b][th][m] = rave[b][th][m].chiral_extr();
                    rave_chir[b][th][m].plot("rave_chir");
                    rave_chir_ave[b][th][m] = rave_chir[b][th][m].average_equiv_moms();
                    rave_chir_ave[b][th][m].plot("rave_chir_ave");
                    
                    /*  O(a2g2) subtraction  */
                    
                    rave_chir_sub[b][th][m] = rave_chir[b][th][m].subtract();
                    rave_chir_sub[b][th][m].plot("rave_chir_sub");
                    rave_chir_sub_ave[b][th][m] = rave_chir_sub[b][th][m].average_equiv_moms();
                    rave_chir_sub_ave[b][th][m].plot("rave_chir_sub_ave");
                }
                
            } //close nm_sea
            
            if(nm_Sea[b]>1 and only_basic==0)
            {
                /*  sea chiral extr  */
                
                rave_chir_sub_sea[b][th] = chiral_sea_extr(rave_chir_sub[b][th]);
                rave_chir_sub_sea[b][th].plot(theta_label[th]+"/rave_chir_sub_sea");
                rave_chir_sub_sea_ave[b][th] = rave_chir_sub_sea[b][th].average_equiv_moms();
                rave_chir_sub_sea_ave[b][th].plot(theta_label[th]+"/rave_chir_sub_sea_ave");
            }
            else if(only_basic==0)
            {
                rave_chir_sub_sea[b][th] = rave_chir_sub[b][th][0];
                rave_chir_sub_sea_ave[b][th] = rave_chir_sub_sea[b][th].average_equiv_moms();
            }
        } //close ntheta
        
        if(ntheta>1 and only_basic==0)
        {
            /*  theta average  */
            
            rave_chir_sub_sea_theta[b] = theta_average(rave_chir_sub_sea[b]);
            rave_chir_sub_sea_theta[b].plot("rave_chir_sub_sea_theta");
            rave_chir_sub_sea_theta_ave[b] = rave_chir_sub_sea_theta[b].average_equiv_moms();
            rave_chir_sub_sea_theta_ave[b].plot("rave_chir_sub_sea_theta_ave");
            
            /*  evolution to 1/a  */
            
            rave_chir_sub_sea_theta_evo[b] = rave_chir_sub_sea_theta[b].evolve(b);
            rave_chir_sub_sea_theta_evo[b].plot("rave_chir_sub_sea_theta_evo");
            rave_chir_sub_sea_theta_evo_ave[b] = rave_chir_sub_sea_theta_evo[b].average_equiv_moms();
            rave_chir_sub_sea_theta_evo_ave[b].plot("rave_chir_sub_sea_theta_evo_ave");
        }
    } //close nbeta
    
    if(nbeta>1 and only_basic==0)
    {
        /*  a2p2->0 extrapolation  */
        
        rave_chir_sub_sea_theta_evo_ave_extr = a2p2_extr(rave_chir_sub_sea_theta_evo_ave);
        
        /*  continuum limit  */
        
        cout<<"Continuum limit extrapolation:"<<endl<<endl;
//        a2p2_extr(rave_chir_sub_sea_theta_evo_ave,LO);
//        a2p2_extr(rave_chir_sub_sea_theta_evo_ave,EM);
    }
    
    cout<<"DONE!"<<endl;
    
    return 0;
}
