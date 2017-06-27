/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef TSNE_H
#define TSNE_H

#include <string>
#include <cassert>
#include <iostream>

using namespace std;

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

//bool stob(string s){
//    if(s.compare("true")==0){
//        return true;
//    }
//    if(s.compare("false")==0){
//        return false;
//    }
//    assert(false);
//}

struct tsne_parameters{
    
    double perplexity,early_exag,eta,theta,min_grad_norm;
    int n_dims,rand_seed,max_iter,n_iter_without_progress,stop_lying_iter,mom_switch_iter,verbose,n_sample,n_feature;
    string file_path;
    
    tsne_parameters(){
        // Default parameters
        n_dims=2;
        perplexity=40.0;
        early_exag=4.0;
        eta=1000; // learning rate
        theta=0.5; // barnes-hut angle
        rand_seed=0;
        max_iter=1000;
        n_iter_without_progress=100;
        min_grad_norm=1e-7;
        stop_lying_iter=250;
        mom_switch_iter=250;
        verbose=1;
        n_sample=0;
        n_feature=0;
        file_path="";
    }
    
    tsne_parameters(char* argv[]){
        update(argv);
    }
    
    void update(char* argv[]){
        n_dims=stoi(argv[1]);
        perplexity=stof(argv[2]);
        early_exag=stof(argv[3]);
        eta=stof(argv[4]);
        theta=stof(argv[5]);
        rand_seed=stoi(argv[6]);
        max_iter=stoi(argv[7]);
        n_iter_without_progress=stoi(argv[8]);
        min_grad_norm=stof(argv[9]);
        stop_lying_iter=stoi(argv[10]);
        mom_switch_iter=stoi(argv[11]);
        verbose=stoi(argv[12]);
        n_sample=stoi(argv[13]);
        n_feature=stoi(argv[14]);
        file_path=argv[15];
    }
    
    void print(){
        cout << "--- t-SNE parameters ---" << endl;
        printf("Embedding dimension\t\t%i",n_dims);
        printf("Perplexity\t\t\t%.2f\n",perplexity);
        printf("Early exaggeration\t\t%.2f\n",early_exag);
        printf("Learning rate\t\t\t%.2f\n",eta);
        printf("Barnes-Hut angle\t\t%.2f\n",theta);
        printf("Random seed\t\t\t%i\n",rand_seed);
        printf("Max iteration\t\t\t%i\n",max_iter);
        printf("Number of iter w/o prog\t\t%i\n",n_iter_without_progress);
        printf("Min grad norm\t\t\t%.3e\n",min_grad_norm);
        printf("Lie switch\t\t\t%i\n",stop_lying_iter);
        printf("Momentum switch\t\t\t%i\n",mom_switch_iter);
        printf("Verbose on ?\t\t\t%i\n",verbose);
        printf("Number of samples\t\t\t%i\n",n_sample);
        printf("Number of features\t\t\t%i\n",n_feature);
    }
};

class TSNE
{
public:
    void run(double* X, int N, int D, double* Y, int no_dims, string file_path, tsne_parameters param=tsne_parameters());
    bool load_data(double** data, int n, int d, string file_path);
    void save_data(double* data, int n, int d, string file_path);
    void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, double** val_P, int N); // should be static!

private:
    void computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
    void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
    double evaluateError(double* P, double* Y, int N, int D);
    double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta);
    void zeroMean(double* X, int N, int D);
    void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity);
    void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);
    void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
    double randn();
};

#endif
