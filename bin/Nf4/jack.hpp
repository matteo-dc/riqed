#ifndef JACK_HPP
#define JACK_HPP

#include "aliases.hpp"

jprop_t clusterize(jprop_t &jS,vvprop_t &S);

vvd_t jackknife(vvd_t &jd, const int size);

jprop_t jackknife(jprop_t &jS);

jvert_t jackknife(jvert_t &jVert);


#endif