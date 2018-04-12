#ifndef MESLEP_HPP
#define MESLEP_HPP

namespace meslep
{
    const int nmeslep = 10;
    const int nprmeslep = 8;
    const int nOp = 5;
    const int nGamma = 11; // number of independent combinations of gamma matrices
    
    const vector<size_t>         iG            = { 1, 2, 3, 4,   0,  10,11,12,13,14,15};
    const vector<int>            g5L_sign      = {-1,-1,-1,-1,  +1,  +1,+1,+1,+1,+1,+1};

    
    const vector<vector<size_t>> iG_of_iop     = {{0,1,2,3},{0,1,2,3},{4},{4},{5,6,7,8,9,10}};
    
    const vector<int>            g5_sign       = { -1,  +1,  -1,  +1,  +1};
    
//    const vector<size_t>         iG            ={1,2,3,4,1,2,3,4,0,0,10,11,12,13,14,15};
//    const vector<int>            g5_sign       ={ -1,-1,-1,-1,  +1,+1,+1,+1,  -1,  +1,  +1,+1,+1,+1,+1,+1};
//    const vector<int>            g5L_sign      ={ -1,-1,-1,-1,  -1,-1,-1,-1,  +1,  +1,  +1,+1,+1,+1,+1,+1};
    
    //    const vector<vector<size_t>> iG_of_iop = {{0,1,2,3},{4,5,6,7},{8},{9},{10,11,12}};
//    const vector<vector<size_t>> iG_of_iop = {{0,1,2,3},{4,5,6,7},{8},{9},{10,11,12,13,14,15}};
    
    const vector<int> proj_norm = {4,4,1,1,24};
    const vector<int> op_norm = {1,1,1,1,2};
    
    enum{_LO=0 ,_F=1 ,_FF=2 ,_T=3 ,_P=4 ,_S=5 };
}

void build_meslep(const vvvprop_t &S1,const vvvprop_t &S2, const vvprop_t &L, valarray<jmeslep_t> &jmeslep);

jvproj_meslep_t compute_pr_meslep(vvvprop_t &jprop1_inv, valarray<jmeslep_t> &jmeslep, vvvprop_t  &jprop2_inv, const double q1, const double q2, const double ql);

#endif