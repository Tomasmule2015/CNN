#include "head.h"
#define af  0.5

class Layer{
private:
	int numofbefore;
	int numofsize;
	arma::fmat aOmg;        //the out put matrix of this layer
	arma::fmat aWmg;         // the weight of the before layer and this layer
	arma::fmat ab;            // the threshold value of this layer
	arma::fmat aBack;         // feedback to before layer;
public:
	Layer(){}
	Layer(int Nofbefore, int Size){
		aWmg = arma::randu<arma::fmat>(Nofbefore, Size);
		aWmg /= 100;
		aOmg.zeros();
		ab = arma::randu<arma::fmat>(1,Size);
		ab /= 100;
		numofbefore = Nofbefore;
		numofsize = Size;
	}
	// return something that net class can use
	int GetSize(){
		return numofsize;
	}
	arma::fmat*  agetoutput(){
		return &aOmg;
	}
	arma::fmat* agetback(){
		return &aBack;
	}
	arma::fmat* agetab(){
		return &ab;
	}
	arma::fmat* agetWmg(){
		return &aWmg;
	}
	
	// BATCHING is batch the source date into 60000/BATCHING part, and we train BATCHING picture update the weight of each layer
	void Compute1(arma::fmat* date);     //date is the front layer's output, it's size BATCHING*numofbefore;
		
	//set data of the first layer;
	void asetStart(int t){
		aOmg = trainim.at(t);
	}
	void asetTStart(int t){
		aOmg = testim.at(t);
	}
	void aUpdateWofend(arma::fmat label, arma::fmat& out);
	void aUpdateW(arma::fmat& out, arma::fmat& back);
	void Load(ifstream &file);
	void Save(ofstream &s);
};
void Layer::Compute1(arma::fmat* date){      //date is the front layer's output, it's size BATCHING*numofbefore;
		aOmg = (*date)*aWmg;
		for (int i = 0; i <BATCHING; i++){
			aOmg.row(i) = aOmg.row(i) + ab;
		}
		aOmg.transform([](float val){return 1.0 / (1 + exp(-val)); });
	}
void Layer::aUpdateWofend(arma::fmat label, arma::fmat& out){   //this fuction just effect among the last neurons layer and output layer; 
		arma::fmat lp;
		lp.set_size(BATCHING, CLASS);
		lp.zeros();    //60*10
		arma::fmat one;
		one.set_size(BATCHING, CLASS);
		one.ones();
		for (int i = 0; i < BATCHING; i++){             //lp save the result of each input BATCHING. 
			lp.at(i, label.at(i, 0)) = 1;
		}

		ep += fabs(accu(aOmg - lp));             //error rate. just used to watch if each ep is down, if not, you network have some problem.

		arma::fmat bak;                           //the output layer's feedback;
		bak.set_size(BATCHING, CLASS);    //60*10
		bak.zeros();
		bak = (aOmg - lp) % (one - aOmg) % aOmg;
		//aOmg.print();
		//bak.print();
		arma::fmat t;
		t.set_size(BATCHING, numofbefore);
		t.zeros();
		t = bak*(aWmg.t());
		arma::fmat tone;
		tone.set_size(BATCHING, numofbefore);
		tone.ones();

		aBack.set_size(BATCHING, numofbefore);       //aBack record the before feedback;
		aBack = t%out % (tone - out);

		//update weight and threshold value
		aWmg -= af / BATCHING*out.t()*bak; 

		ab -= af/BATCHING*sum(bak,0);
		
	}
void Layer::aUpdateW(arma::fmat& out,arma::fmat&back){
        arma::fmat one;
		one.set_size(BATCHING, numofbefore);
		one.ones();

		arma::fmat t;
		t.set_size(BATCHING, numofbefore);
		t.zeros();
		t = back*(aWmg.t());

		arma::fmat tone;
		tone.set_size(BATCHING, numofbefore);
		tone.ones();
	
		aBack.set_size(BATCHING, numofbefore);
		aBack = t%out % (tone - out);
		//t.print();
		aWmg -= af / BATCHING*out.t()*back;
		ab -= af / BATCHING*sum(back,0);
}
void Layer::Load(ifstream &file){
        (file)>>numofbefore>>numofsize;
		string matrix[4];
		for(int i =0; i < 4;i++){
		    (file)>>matrix[i];
		}
	    cout<<"szie:" <<numofbefore<<"*"<<numofsize<<'\n';
		aOmg.load(matrix[0]);
		aWmg.load(matrix[1]);
		ab.load(matrix[2]);
		aBack.load(matrix[3]);
}
void Layer::Save(ofstream &s){
		char p[10];
		char q[10];
		itoa(numofsize,p,10);
		itoa(numofbefore,q,10);
		string qq(q);
		string pp(p);
		cout<<"szie:" <<numofbefore<<"*"<<numofsize<<'\n';
		string s1 = qq+"."+pp+"aOmg.txt";
		string s2 = qq+"."+pp+"aWmg.txt";
		string s3 = qq+"."+pp+"ab.txt";
		string s4 = qq+"."+pp+"aBack.txt";
		s<<numofbefore<<'\n'<<numofsize<<'\n';
		s<<s1<<'\n';
		s<<s2<<'\n';
	 	s<<s3<<'\n';
	    s<<s4<<'\n';
		aOmg.save(s1);
		aWmg.save(s2);
		ab.save(s3);
		aBack.save(s4);
}

