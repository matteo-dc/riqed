#ifndef _AVE_ERR_HPP
#define _AVE_ERR_HPP

#include <tuple>
#include "aliases.hpp"

// average of bilinears and Z
tuple<vvvvd_t,vvvvd_t> ave_err(vector<jproj_t> jG);

// average of Zq
tuple<vvd_t,vvd_t> ave_err(vector<vvd_t> jZq);

// average of effective mass
tuple<vvd_t,vvd_t> ave_err(vvvd_t jM);


#endif