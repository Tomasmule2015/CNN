/*
   Author:  MAYIHUI
   Date: 2015/7/15
   Email: benze_ma@163.com
   brief introduction of this project:
         It's a project about Stanford handwrittten numeral distinguish, there is a depth network, you can add layer and change input what you want.
		 writed by C++.

   enviroonment support: opencv and armadillo.I used opencv to read binary file,and armadillo matix operation.
*/

#include"net.h"
#include<string>
using namespace std;

int main(int argc, char** argv) {

	const string filename = "train-images.idx3-ubyte";
	const string labelname = "train-labels.idx1-ubyte";      
	
	cout<<"You can load the last save network,if you input 'y'";
	char p ;
	cin>>p;


	//  filename is trian file's name, labelname is label file's name.
	                                                       //3 is about this net work is 4 layer,include input layer, two neurons layer, and output layer
	                                                       //361 is the col of each image. why is not 784, since I get ROI of each picture, and is not a matix 28*28, is 19*19;
	                                                       //100 is the first neurons number, the after neurons number = 100 - (firstnerous - CLASS) / hide; this hide = 3;
	if(p == 'y'){
		Net* net = new  Net(filename, labelname,784);
		net->Load();
	    net->CTest();        //test;
	}else{
		Net* net = new  Net(filename, labelname, 2, 784,500);  
		net->Train(1);     //this 100 train 100 times; one train include 60000 picture;
	    net->CTest();        //test;
	}
	return 0;
}