#ifndef READ_HPP
#define READ_HPP

#include "aliases.hpp"
#include <fstream>
#include <iostream>
#include <vector>

void read_mom_list(const string &path);

double read_plaquette();

vvd_t read_eff_mass(const string name);

vvvd_t read_deltam_cr(const string name);

size_t isc(size_t is,size_t ic);

string path_to_conf(const string &string_path, int i_conf,const string &name);

vector<string> setup_read_prop(ifstream input[]);

prop_t read_prop(ifstream &input, const string &path, const int imom);

vvvprop_t read_prop_mom(ifstream input[],const vector<string> v_path,const int i_in_clust,const int ihit,const int imom);

void read_internal(double &t,ifstream& infile);

void read_internal(VectorXd &V, ifstream& infile);

template <class T>
void read_internal(valarray<T> &v, ifstream& infile)
{
    for(auto &i : v) read_internal(i,infile);
}

template <class T>
void read_vec( T &vec, const char* path)
{
    ifstream infile;
    infile.open(path,ifstream::in);
    
    if (infile.is_open())
    {
        for(auto &i : vec)
            read_internal(i,infile);
        
        infile.close();
        
    }
    else cout << "Unable to open the input file "<<path<<endl;
}

#define READ(NAME)				\
read_vec(NAME,"print/"#NAME".txt");

#define READ2(NAME1,NAME2)       \
read_vec(NAME1,"print/"#NAME2".txt");


#endif
