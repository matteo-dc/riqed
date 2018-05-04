#ifndef _AVE_ERR_HPP
#define _AVE_ERR_HPP

#include <tuple>
#include "aliases.hpp"

// average sigma
tuple<vvvd_t,vvvd_t> ave_err(vvvvd_t sigma);

// average of Zq
tuple<vvd_t,vvd_t> ave_err(vector<vvd_t> jZq);

// average of bilinears
tuple<vvvvvd_t,vvvvvd_t> ave_err(vector<jproj_t> jG);

// average of Zbil
tuple<vvvvd_t,vvvvd_t> ave_err(vector<jZbil_t> jZ);

// average of effective mass
tuple<vvd_t,vvd_t> ave_err(vvvd_t jM);

// average of deltam
tuple<vd_t,vd_t> ave_err(vvd_t jdeltam);

// average meslep and Z4f
tuple<vvvvvd_t,vvvvvd_t> ave_err(vector<jproj_meslep_t> jZ4f);



#endif