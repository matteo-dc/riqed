#include "aliases.hpp"
#include "global.hpp"
#include "Dirac.hpp"
#include <omp.h>
#include "operations.hpp"

void oper_t::compute_Zq()
{
    for(int ilinmom=0; ilinmom<_linmoms; ilinmom++)
    {
        // LO
        jZq[ilinmom] = sigma1_LO[ilinmom];
        
        // EM
        jZq_EM[ilinmom] = sigma1_PH[ilinmom] +
                          sigma1_P[ilinmom]*deltam_cr +
                          sigma1_S[ilinmom]*deltamu;
    }
}