#ifndef _CONTR_HPP
#define _CONTR_HPP

#include "global.hpp"
#include "aliases.hpp"

const double tau3[2]={-1.0,+1.0}; //!< tau entering the propagator

enum                      {_LO ,_F ,_FF ,_T ,_P ,_S ,_QED=2};

enum{RE,IM};

const int ODD=-1,UNK=0,EVN=1;

dcompl coeff_to_read(const size_t ikind,const size_t r);

//vvd_t get_contraction(const string &suffix, const string &out, const int mr1, const string &T1, const int mr2, const string &T2, const string &ID, const string &reim, const string &parity, const int* conf_id, const string &string_path);

vvd_t get_contraction(const string &suffix, const string &out, const int m_fw, const int m_bw, const int rfw, const int rbw,size_t kfw, size_t kbw, const string &ID, const size_t ext_reim, const int &tpar, const int* conf_id , const string &string_path);

#endif