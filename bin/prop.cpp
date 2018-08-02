#include "global.hpp"
#include "aliases.hpp"
#include "operations.hpp"

#define EXTERN_PROP
 #include "prop.hpp"

namespace qprop
{
    void set_ins()
    {
        if(ntypes==6) ins_list={LO,F,FF,T,P,S};
        if(ntypes==3) ins_list={LO,F,QED};
        nins=ins_list.size();
    }
}
namespace jprop
{
    void set_ins()
    {
        if(ntypes==6) ins_list={LO,PH,P,S};
        if(ntypes==3) ins_list={LO,QED};
        nins=ins_list.size();
    }
}
namespace lprop
{
    void set_ins()
    {
        ins_list={LO,F};
        nins=ins_list.size();
    }
}

void oper_t::build_prop(const vvvprop_t &prop, vvvprop_t &jprop)
{
#pragma omp parallel for collapse(2)
    for(int mr=0;mr<nmr;mr++)
        for(int ijack=0;ijack<njacks;ijack++)
        {
            if(ntypes==6)
            {
                jprop[jprop::LO][ijack][mr]  += prop[ijack][qprop::LO][mr];  // Leading order
                jprop[jprop::PH][ijack][mr]  += prop[ijack][qprop::FF][mr] +
                                                prop[ijack][qprop::T][mr];   // self energy + tadpole
                jprop[jprop::P][ijack][mr]   += prop[ijack][qprop::P][mr];   // P insertion
                jprop[jprop::S][ijack][mr]   += prop[ijack][qprop::S][mr];   // S insertion
            }
            if(ntypes==3)
            {
                jprop[jprop::LO ][ijack][mr] += prop[ijack][qprop::LO ][mr]; // Leading order
                jprop[jprop::QED][ijack][mr] += prop[ijack][qprop::QED][mr]; // QED correction
            }
            
        }
}


// invert the propagator
vvprop_t invert_jprop(const vvprop_t &jprop)
{
    vvprop_t jprop_inv(valarray<prop_t>(prop_t::Zero(),nmr),njacks);
    
#pragma omp parallel for collapse(2)
    for(int ijack=0;ijack<njacks;ijack++)
        for(int mr=0;mr<nmr;mr++)
            jprop_inv[ijack][mr]=jprop[ijack][mr].inverse();
    
    return jprop_inv;
}