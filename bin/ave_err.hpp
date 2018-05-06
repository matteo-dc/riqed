#ifndef _AVE_ERR_HPP
#define _AVE_ERR_HPP

#include <tuple>
#include "aliases.hpp"

void get_sizes(double vec, vector<size_t> &sizes);

template <class T>
void get_sizes(T vec, vector<size_t> &sizes)
{
    sizes.push_back(vec.size());
    cout<<vec.size()<<endl;
    
    get_sizes(vec[0],sizes);
}

template <class T>
void get_sizes(vector<T> vec, vector<size_t> &sizes)
{
    sizes.push_back(vec.size());
    cout<<vec.size()<<endl;
    
    get_sizes(vec[0],sizes);
}

template <class T>
tuple<T,T> ave_err(vector<T> vec)
{
    vector<size_t> sizes;
    
    get_sizes(vec,sizes);
}

template <class T>
tuple<T,T> ave_err(valarray<T> vec)
{
    vector<size_t> sizes;
    
    get_sizes(vec,sizes);
}


//// average sigma
//tuple<vvvvd_t,vvvvd_t> ave_err(vector<vvvvd_t> sigma);
//
//// average of Zq
//tuple<vvd_t,vvd_t> ave_err(vector<vvd_t> jZq);
//
//// average of bilinears
//tuple<vvvvvd_t,vvvvvd_t> ave_err(vector<jproj_t> jG);
//
//// average of Zbil
//tuple<vvvvd_t,vvvvd_t> ave_err(vector<jZbil_t> jZ);
//
//// average of effective mass
//tuple<vvd_t,vvd_t> ave_err(vvvd_t jM);
//
//// average of deltam
//tuple<vd_t,vd_t> ave_err(vvd_t jdeltam);
//
//// average meslep
//tuple<vvvvvvd_t,vvvvvvd_t> ave_err(vector<jproj_meslep_t> jmeslep);
//
//// average Z4f
//tuple<vvvvvd_t,vvvvvd_t> ave_err_Z4f(vector<jZ4f_t> jZ4f);
//


#endif