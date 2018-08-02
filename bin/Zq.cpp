#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <omp.h>
#include "operations.hpp"
#include "sigmas.hpp"

using namespace sigma;

void oper_t::compute_Zq()
{
    cout<<"Computing Zq"<<endl;
    
#pragma omp parallel for collapse(3)
    for(int ilinmom=0; ilinmom<_linmoms; ilinmom++)
        for(int ijack=0;ijack<njacks;ijack++)
            for(int mr=0;mr<_nmr;mr++)
            {
                // LO
                jZq[ilinmom][ijack][mr] = sigma[ilinmom][SIGMA1][LO][ijack][mr];
                
                // EM (relative)
                jZq_EM[ilinmom][ijack][mr] =
                    sigma[ilinmom][SIGMA1][PH][ijack][mr] /
                    sigma[ilinmom][SIGMA1][LO][ijack][mr];
                if(ntypes==6)
                    jZq_EM[ilinmom][ijack][mr] +=
                        (sigma[ilinmom][SIGMA1][P ][ijack][mr]*deltam_cr[ijack][mr] +
                         sigma[ilinmom][SIGMA1][S ][ijack][mr]*deltamu[ijack][mr]) /
                        sigma[ilinmom][SIGMA1][LO][ijack][mr];
            }
}