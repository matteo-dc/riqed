#ifndef READ_HPP
#define READ_HPP

#include "aliases.hpp"
#include <vector>

void read_mom_list(const string &path);

vector<vector<double>> read_eff_mass(const string name,const int nmr);

vvvd_t read_deltam_cr(const string name, const int nm, const int njacks);

size_t isc(size_t is,size_t ic);

string path_to_conf(const string &string_path, int i_conf,const string &name);

vector<string> setup_read_prop(ifstream input[]);

prop_t read_prop(ifstream &input, const string &path, const int imom);

vvvprop_t read_prop_mom(ifstream input[],const vector<string> v_path,const int i_in_clust,const int ihit,const int imom);


#endif
