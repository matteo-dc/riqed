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
    for(int ilinmom=0; ilinmom<_linmoms; ilinmom++)
    {
        // LO
        jZq[ilinmom] = sigma[ilinmom][SIGMA1][LO];
        
        // EM (relative)
        jZq_EM[ilinmom] = (sigma[ilinmom][SIGMA1][PH] +
                           sigma[ilinmom][SIGMA1][P ]*deltam_cr +
                           sigma[ilinmom][SIGMA1][S ]*deltamu)/
                            sigma[ilinmom][SIGMA1][LO];
    }
}