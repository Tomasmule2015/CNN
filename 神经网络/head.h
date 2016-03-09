#include<iostream>
#include<vector>
#include<math.h>
#include<windows.h>
#include<time.h>
#include <opencv2\opencv.hpp>
#include <armadillo>
using namespace std;
using namespace cv;

#define BATCHING 1
#define CLASS 10
#define SAVETIME 20
double ep;

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