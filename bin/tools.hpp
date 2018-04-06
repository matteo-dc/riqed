#ifndef _TOOLS_HPP
#define _TOOLS_HPP

#include "aliases.hpp"

//! take the forward derivative
template <class T> T forward_derivative(const T &v)
{
  const size_t n=v.size();
  T out(n);
  for(size_t it=0;it<n;it++) out[it]=v[(it+1)%n]-v[it];
  return out;
}

//! take the symmetric derivative
template <class T> T symmetric_derivative(const T &v)
{
  const size_t n=v.size();
  T out(n);
  for(size_t it=0;it<n;it++) out[it]=(v[(it+1)%n]-v[(it-1+n)%n])/2.0;
  return out;
}

//! return the symmetric
vd_t symmetric(const vd_t &data);

//! return the averaged
vd_t symmetrize(const vd_t &data, int par=1);

#endif
