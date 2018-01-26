#ifndef INPUT_HPP
#define INPUT_HPP

#include "global.hpp"
#include "aliases.hpp"

// parse the value string
template <class T>
void get_value(FILE *fin,T &ret,const char *t);


// check
void check_str_par(const char *str,const char *name);
void check_int_par(const int val,const char *name);
void check_double_par(const double val,const char *name);

// read input file
void read_input(const char input_path[]);

#endif
