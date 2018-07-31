#ifndef SIGMAS_HPP
#define SIGMAS_HPP

#include "aliases.hpp"
#include "global.hpp"

#ifndef EXTERN_SIGMA
 #define EXTERN_SIGMA extern
#endif

namespace sigma
{
    void set_ins();
    
    enum proj{SIGMA1,SIGMA2,SIGMA3};
    enum ins{LO,PH,P,S,QED=1};
    
    EXTERN_SIGMA vector<proj> proj_list;
    EXTERN_SIGMA vector<ins>  ins_list;
    
    EXTERN_SIGMA int nproj;
    EXTERN_SIGMA int nins;
    EXTERN_SIGMA int nsigma;
    
    EXTERN_SIGMA vector<string> proj_tag;
    EXTERN_SIGMA vector<string> ins_tag;
}

#undef EXTERN_SIGMA

#endif