#ifndef VERT_HPP
#define VERT_HPP

#include "aliases.hpp"

// compute LO and EM vertices
void build_vert(const vvvprop_t &S1,const vvvprop_t &S2,valarray<jvert_t> &jVert_LO_and_EM);

// calculate the vertex function in a given configuration for the given equal momenta
//prop_t make_vertex(const prop_t &prop1, const prop_t &prop2, const int mu);

#endif 