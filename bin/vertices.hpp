#ifndef VERT_HPP
#define VERT_HPP

#include "aliases.hpp"

#ifndef EXTERN_VERT
 #define EXTERN_VERT extern
#endif

// compute LO and EM vertices
void build_vert(const vvvprop_t &S1,const vvvprop_t &S2,valarray<jvert_t> &jVert);

namespace gbil
{
    void set_ins();
    
    enum ins{LO,PH,Pfw,Pbw,Sfw,Sbw,QED=1};
    EXTERN_VERT vector<ins> ins_list;
    EXTERN_VERT int nins;
    
    EXTERN_VERT int nGamma;
    
    EXTERN_VERT vector<string> ins_tag;
}

#undef EXTERN_VERT

#endif 