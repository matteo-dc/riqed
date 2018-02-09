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

//void print_internal_bin(int t,ofstream& outfile);
//void print_internal_bin(double t,ofstream& outfile);

template <class T>
void print_internal_bin(T t,ofstream& outfile)
{
    outfile.write((char*) &t,sizeof(T));
}

void print_internal_bin(VectorXd &V, ofstream& outfile);

template <class T>
void print_internal_bin(valarray<T> &v, ofstream& outfile)
{
    for(auto &i : v) print_internal_bin(i,outfile);
}


template <class T>
void print_vec( T &vec, const string path)
{
//    ofstream outfile(path/*,ofstream::binary*/);
    ofstream outfile;
    outfile.open(path.c_str());
    
    if (outfile.is_open())
    {
        for(auto &i : vec)
            print_internal(i,outfile);
        
        outfile.close();
        
    }
    else cout << "Unable to open the output file "<<path<<endl;
}

template <class T>
void print_vec_bin( T &vec, const string path)
{
    //    ofstream outfile(path/*,ofstream::binary*/);
    ofstream outfile;
    outfile.open(path.c_str(), ios::binary);
    
    if (outfile.is_open())
    {
        for(auto &i : vec)
            print_internal_bin(i,outfile);
        
        outfile.close();
        
    }
    else cout << "Unable to open the output file "<<path<<endl;
}


#define PRINT(NAME)				\
print_vec(NAME,"print/"#NAME".txt");

#define PRINT_BIN(NAME)         \
print_vec_bin(NAME,"print/"#NAME);


#endif
