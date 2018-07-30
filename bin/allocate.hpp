#ifndef ALLOC_HPP
#define ALLOC_HPP

#include "global.hpp"
#include "aliases.hpp"

#include "operations.hpp"

void allocate_vec_internal(double &t, const vector<int> sizes, int isize);

void allocate_vec_internal(oper_t &o, const vector<int> sizes, int isize);

template <class T>
void allocate_vec_internal(valarray<T> &v, const vector<int> sizes, int isize)
{
    v.resize(sizes[isize]);
    isize++;
    for(auto &i : v)
        allocate_vec_internal(i,sizes,isize);
}

template <class T>
void allocate_vec( T &vec, const vector<int> sizes)
{
    int isize=0;
    
    vec.resize(sizes[isize]);
    isize++;
    
#pragma omp parallel for
    for(size_t i=0;i<vec.size();i++)
        allocate_vec_internal(vec[i],sizes,isize);

//    for(auto &i : vec)
//        allocate_vec_internal(i,sizes,isize);
}

#endif