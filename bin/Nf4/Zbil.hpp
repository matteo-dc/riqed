#ifndef _ZBIL_HPP
#define _ZBIL_HPP

#include "aliases.hpp"

//jproj_t compute_pr_bil( jprop_t &jS1_inv,  jvert_t &jVert,  jprop_t  &jS2_inv);

vvvvvd_t compute_pr_bil( vvvprop_t &jprop1_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jprop2_inv);

#endif