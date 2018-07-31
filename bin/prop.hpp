#ifndef PROP_HPP
#define PROP_HPP

#ifndef EXTERN_PROP
 #define EXTERN_PROP extern
#endif

namespace jprop
{
    void set_ins();
    
    enum ins{LO,PH,P,S,QED=1};
    
    EXTERN_PROP vector<ins> ins_list;
    EXTERN_PROP int nins;
}

namespace qprop
{
    void set_ins();
    
    enum ins{LO,F,FF,T,P,S,QED=2};
    
    EXTERN_PROP vector<ins> ins_list;
    EXTERN_PROP int nins;
}

namespace lprop
{
    void set_ins();
    
    enum ins{LO,F};
    
    EXTERN_PROP vector<ins> ins_list;
    EXTERN_PROP int nins;
}

// invert the propagator
vvprop_t invert_jprop( const vvprop_t &jprop);

#undef EXTERN_PROP

#endif