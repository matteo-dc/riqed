#ifndef _ZBIL_HPP
#define _ZBIL_HPP

#include "aliases.hpp"

vvvvvd_t compute_pr_bil( vvvprop_t &jprop1_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jprop2_inv);

vvvvvd_t compute_pr_bil_4f( vvvprop_t &jprop1_inv,  valarray<jvert_t> &jVert,  vvvprop_t  &jprop2_inv, const double q1, const double q2);


#endif