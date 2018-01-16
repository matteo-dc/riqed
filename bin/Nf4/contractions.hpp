#ifndef _CONTR_HPP
#define _CONTR_HPP

string path_to_contr(int i_conf, const int mr1, const string &T1, const int mr2, const string &T2);

vvd_t get_contraction(const int mr1, const string &T1, const int mr2, const string &T2, const string &ID, const string &reim, const string &parity, const int* conf_id);

#endif