#define _CRTDBG_MAP_ALLOC
#include<stdlib.h>
#include<crtdbg.h>
#include "Cnet.h"




int main(){
	Cnet net;
	//net.~Cnet
	cout<<"Please input 'n' to not load net:\n";
	char p;
	cin>>p;
	if('n' == p)
	{
		net.Trian(10);
		
	}else{
		net.Load(0);
		int i = 0;
	
		//net.Trian(10);
		net.CTest(1);
	   
	}
	//Net* net = new  Net(2, 784, 500);
	//net->ReadDate();
	//net->ReadTest();
	//net->date();
	//net->Train(60);     //this 100 train 100 times; one train include 60000 picture;
	//net->CTest();        //test;
	im.clear();
	for(vector<arma::fmat*>::iterator s = traininput.begin();s!=traininput.end();s++){
	   delete *s;
	}
	_CrtDumpMemoryLeaks();
	return 0;
	
	
}