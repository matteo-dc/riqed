#ifndef READ_HPP
#define READ_HPP

#include "aliases.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

double read_plaquette(const string &path);

string path_to_conf(const string &string_path, int i_conf,const string &name);

vector<string> setup_read_qprop(FILE* input[]);

size_t isc(size_t is,size_t ic);

vvvprop_t read_qprop_mom(FILE* input[],const vector<string> v_path,const int i_in_clust,const int ihit,const int imom);

vvprop_t read_lprop_mom(FILE* input[],const vector<string> v_path,const int i_in_clust,const int ihit,const int imom);


// to string with precision
template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
    ostringstream out;
    out << fixed;
    out << setprecision(n) << a_value;
    return out.str();
}

void read_internal(double &t,ifstream& infile);
void read_internal(VectorXd &V, ifstream& infile);

template <class T>
void read_internal(valarray<T> &v, ifstream& infile)
{
    for(auto &i : v) read_internal(i,infile);
}

template <class T>
void read_internal_bin(T &t,ifstream& infile)
{
    infile.read((char*) &t,sizeof(T));
}

template <class T>
void read_internal_bin(valarray<T> &v, ifstream& infile)
{
    for(auto &i : v) read_internal_bin(i,infile);
}

template <class T>
void read_vec( T &vec, const string path)
{
    ifstream infile;
    infile.open(path.c_str(),ifstream::in);
    
    if (infile.is_open())
    {
        for(auto &i : vec)
            read_internal(i,infile);
        
        infile.close();
        
    }
    else cout << "Unable to open the input file "<<path<<endl;
}

template <class T>
void read_vec_bin( T &vec, const string path)
{
    ifstream infile;
    infile.open(path.c_str(),ios::binary);
    
    if (infile.is_open())
    {
        for(auto &i : vec)
            read_internal_bin(i,infile);
        
        infile.close();
        
    }
    else cout << "Unable to open the input file "<<path<<endl;
}

#define READ(NAME)				\
read_vec(NAME,"print/"#NAME".txt");

#define READ2(NAME1,NAME2)       \
read_vec(NAME1,"print/"#NAME2".txt");

#define READ_BIN(NAME)           \
read_vec_bin(NAME,"print/"#NAME);


#endif
