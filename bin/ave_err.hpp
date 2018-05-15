#ifndef _AVE_ERR_HPP
#define _AVE_ERR_HPP

#include <tuple>
#include "aliases.hpp"

//template <class T>
//void compute_ave( T &ave, T &sqr_ave, vector<T> vec)
//{
//    for(int i=0; i<vec.size(); i++)
//        compute_ave_internal();
//}
//
//
//void get_sizes(double vec, vector<size_t> &sizes);
//
//template <class T>
//void get_sizes(valarray<T> vec, vector<size_t> &sizes)
//{
//    sizes.push_back(vec.size());
//    cout<<vec.size()<<endl;
//    
//    get_sizes(vec[0],sizes);
//}
//
//template <class T>
//void get_sizes(vector<T> vec, vector<size_t> &sizes)
//{
//    sizes.push_back(vec.size());
//    cout<<vec.size()<<endl;
//    
//    get_sizes(vec[0],sizes);
//}
//
//template <class T, int pos_jack>
//tuple<T,T> ave_err(vector<T> vec)
//{
//    for(int i=0; i<vec.size(); i++)
//        ave_err<pos_jack-1>(vec[i]);
//}
//
//template <class T>
//tuple<T,T> ave_err<T,0>(vector<T> vec)
//{
//    vector<int> sizes, new_sizes;
//    
//    get_sizes(vec,sizes);
//    
//    for(int i=0;i<sizes.size();i++)
//        if(i!=col)
//            new_sizes.push_back(sizes[i]);
//    
//    T ave, sqr_ave, err;
//    
//    allocate_vec(ave,new_sizes);
//    allocate_vec(sqr_ave,new_sizes);
//    allocate_vec(err,new_sizes);
//}
//
//template <class T>///////////////
//tuple<T,T> ave_err(valarray<T> vec, int col)
//{
//    vector<int> sizes;
//    
//    get_sizes(vec,sizes);
//}

// average sigma
tuple<vvvvd_t,vvvvd_t> ave_err(vector<vvvvd_t> sigma);

// average of Zq
tuple<vvd_t,vvd_t> ave_err_Zq(vector<vvd_t> jZq);

// average of bilinears
tuple<vvvvvd_t,vvvvvd_t> ave_err(vector<jproj_t> jG);

// average of Zbil
tuple<vvvvd_t,vvvvd_t> ave_err_Z(vector<jZbil_t> jZ);

// average of valence effective mass and mPCAC
tuple<vvd_t,vvd_t> ave_err(vvvd_t jM);

// average of sea effective mass
tuple<double,double> ave_err(vd_t jM);

// average of deltam
tuple<vd_t,vd_t> ave_err(vvd_t jdeltam);

// average meslep
tuple<vvvvvvd_t,vvvvvvd_t> ave_err(vector<jproj_meslep_t> jmeslep);

// average Z4f
tuple<vvvvvd_t,vvvvvd_t> ave_err_Z4f(vector<jZ4f_t> jZ4f);



#endif