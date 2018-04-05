#ifndef _TOOLS_HPP
#define _TOOLS_HPP

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

#endif
