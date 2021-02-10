#define GCC

#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "tools.h"
// #include "test_matrix.h"
// B (PKT_SIZE,DEGREE)
// G (DEGREE,BATCH_SIZE)

using namespace std;
int main(){
    char buf[100];
    int n = 0;
    srand(SEED-1);
    ofstream myfile;
    myfile.open ("./include/test_matrix.h");


    myfile << "#include \"config.h\"\n\nstatic uint8_t B[PKT_SIZE][DEGREE] = {";
    // print arrays
    // B (PKT_SIZE,DEGREE)
    uint8_t B[PKT_SIZE][DEGREE];
    for(int j=0;j<PKT_SIZE;j++){
        myfile << "\n\t\t{";
        for(int i=0;i<DEGREE;i++){
            uint8_t num = rand() % 256;
            B[j][i] = num;
            n = sprintf(buf,"%d,",num);
            myfile << buf;
        }
        myfile << "},";
    }
    myfile << "\n\t};\n";

    // G (DEGREE,BATCH_SIZE)
    uint8_t G[DEGREE][BATCH_SIZE];
    srand(SEED);
    myfile << "\n\nstatic uint8_t G[DEGREE][BATCH_SIZE] = {";
    for(int j=0;j<DEGREE;j++){
        myfile << "\n\t\t{";
        for(int i=0;i<BATCH_SIZE;i++){
            int num = rand() % 256;
            G[j][i] = num;
            n = sprintf(buf,"%d,",num);
            myfile << buf;
        }
        myfile << "},";
    }
    myfile << "\n\t};\n";
    
   


    // multiply them
    uint8_t B_[PKT_SIZE*DEGREE];
    uint8_t G_[DEGREE*BATCH_SIZE];
    uint8_t res[PKT_SIZE][BATCH_SIZE];
    uint8_t res_[PKT_SIZE*BATCH_SIZE];
    matrix_flatten<PKT_SIZE,DEGREE>(B,B_);
    matrix_flatten<DEGREE,BATCH_SIZE>(G,G_);
    matrix_multi(B_,G_,res_,PKT_SIZE,DEGREE,DEGREE);
    matrix_reform<PKT_SIZE,BATCH_SIZE>(res_,res);

    myfile << "\n\nstatic uint8_t res[PKT_SIZE][BATCH_SIZE] = {";
    for(int j=0;j<PKT_SIZE;j++){
        myfile << "\n\t\t{";
        for(int i=0;i<BATCH_SIZE;i++){
            n = sprintf(buf,"%d,",res[j][i]);
            myfile << buf;
        }
        myfile << "},";
    }
    myfile << "\n\t};\n";

    myfile << "\n\nstatic uint8_t deg_list[N_BATCH] = {";
    for(int j=0;j<N_BATCH;j++){
        // int num = rand() % 64;
        int num = 16;
        while(num % TS != 0 || num ==0){
            num = rand() % 64;
        }
        n = sprintf(buf,"%d,",num);
        myfile << buf;
    }
    myfile << "};\n";



    myfile.close();
    return 0;


}

