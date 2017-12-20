#ifndef _FIT_HPP
#define _FIT_HPP

#include "aliases.hpp"

vvd_t fit_par_jackknife(const vvd_t &coord, const int n_par, vd_t &error, const vvd_t &y, const int range_min, const int range_max);


#endif