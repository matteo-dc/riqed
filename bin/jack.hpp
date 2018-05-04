#ifndef JACK_HPP
#define JACK_HPP

#include "aliases.hpp"

// clusterize propagator
jprop_t clusterize(jprop_t &jS,vvprop_t &S);
// jackknife double
vvd_t jackknife(vvd_t &jd);
// jackknife Propagator
vvprop_t jackknife(vvprop_t &jS);
// jackknife Vertex
jvert_t jackknife(jvert_t &jVert);
// jackknife meslep
jmeslep_t jackknife(jmeslep_t &jmeslep);

#endif