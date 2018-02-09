#ifndef VERT_HPP
#define VERT_HPP

#include "aliases.hpp"

//// compute the LO vertex
//jvert_t build_LO_vert(vvvprop_t &S1,vvvprop_t &S2, jvert_t &jVert_0);
//
//// compute the EM vertex (up to 1st order in QED)
//jvert_t build_EM_vert(vvvprop_t &S1,vvvprop_t &S2,vvprop_t &S1_em,vvprop_t &S2_em,jvert_t &jVert_em);

// compute LO and EM vertices
/*valarray<jvert_t>*/void build_vert(const vvvprop_t &S1,const vvvprop_t &S2,/*const vvprop_t &S1_em,const vvprop_t &S2_em,*/valarray<jvert_t> &jVert_LO_and_EM);


// calculate the vertex function in a given configuration for the given equal momenta
//prop_t make_vertex(const prop_t &prop1, const prop_t &prop2, const int mu);

// invert the propagator
jprop_t invert_jprop( const jprop_t &jprop);

#endif 