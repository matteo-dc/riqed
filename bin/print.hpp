#ifndef PRINT_HPP
#define PRINT_HPP

#include "aliases.hpp"
#include <iostream>
#include <fstream>

void print_internal(double t,ofstream& outfile);

void print_internal(VectorXd &V, ofstream& outfile);

template <class T>
void print_internal(valarray<T> &v, ofstream& outfile)
{
    for(auto &i : v) print_internal(i,outfile);
}

template <class T>
void print_vec( T &vec, const char* path)
{
//    ofstream outfile(path/*,ofstream::binary*/);
    ofstream outfile;
    outfile.open(path);
    
    if (outfile.is_open())
    {
        for(auto &i : vec)
            print_internal(i,outfile);
        
        outfile.close();
        
    }
    else cout << "Unable to open the output file "<<path<<endl;
}

#define PRINT(NAME)				\
print_vec(NAME,"print/"#NAME".txt");


#endif
