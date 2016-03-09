#include <stdlib.h>
#include<iostream>     //malloc
#include<vector>
#include<armadillo>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

#define SAVETIME 10
#define N 2
#define BATCHING 1
#define CLASS 10

struct block{
    arma::fmat amatrix;
	block(){}
	block(int size){
        amatrix = arma::randn<arma::fmat>(size,size);
		amatrix  = amatrix;
		//theate = (rand()%10)/100.0;
	}
	void Set(arma::fmat matrix){amatrix = matrix;}
};
int rate = 0;        //count the right number of size
double ep = 0;
struct IMG{
	double image[784];
	int l_label;
};
vector<IMG> im;
vector<IMG> test;
vector<arma::fmat> trainim;
vector<arma::fmat> trainlabel;
vector<arma::fmat> testim;
vector<arma::fmat> testlabel;
vector<arma::fmat*> traininput;
vector<arma::fmat*> testinput;
arma::fmat train;
arma::fmat testt;