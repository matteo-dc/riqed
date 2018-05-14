#ifndef _FIT_HPP
#define _FIT_HPP

#include "aliases.hpp"

vvd_t polyfit(const vvd_t &coord, const int n_par, vd_t &error, const vvd_t &y, const double xmin, const double xmax);

vvd_t polyfit(const vvd_t &coord, const int n_par, vd_t &error, const vvd_t &y, const int range_min, const int range_max);

vd_t fit_continuum(const vvd_t &coord, vd_t &error, const vd_t &y, const int p2_min, const int p2_max, const double &p_min_value);

vvd_t fit_par(const vvd_t &coord, const vd_t &error, const vvd_t &y, const int range_min, const int range_max/*,const string &path=NULL*/);

vXd_t fit_chiral_jackknife(const vvd_t &coord, vd_t &error, const vector<vd_t> &y, const int range_min, const int range_max, const double &p_min_value);

vvXd_t fit_chiral_jackknife(const vvd_t &coord, vvd_t &error, const vector<vvd_t> &y, const int range_min, const int range_max, const double &p_min_value);

#endif