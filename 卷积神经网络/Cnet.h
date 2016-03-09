#include "net.h"


class Cnet{
     private: 
		 //arma::fmat out;                  //
     	arma::fmat* back;                //return from full conet
     	int batching;
		CLayer* S;
		CLayer* S1;
		Net* net;


     public:
        Cnet(){
		
		S = new CLayer(1,20,5,24);
        S1 = new CLayer(20,50,5,8);
        net = new Net(2,800,500);
        }
	    ~Cnet(){
			S->~CLayer();
			S1->~CLayer();
			
		}
		void Save(int i){
			S->Save(i);
			S1->Save(i);
			net->Save(i);
		}
		void Load(int i){
			S->Load(20,i);
			S1->Load(1000,i);
			net->Load(i);
		}
		void Test(int i){
            S->ComputeF(testinput[i]);          //计算得到12*12 *20矩阵， 存入S presult中， 
		//	traininput.at(i)->print("this is input:\n");
			vector<block*> w = S->Getweight();

			//  w.at(0)->amatrix.print("this is weight of s:\n");
		  //  w.at(10)->amatrix.print("this is weight of s:\n");
			  vector<arma::fmat*> re = S->Getresult();
			//  re.at(0)->print("this is result of S:\n");
			  vector<arma::fmat*> re2 = S->Getpresult();
			//  re2.at(10)->print("this is presult of S:\n");
			  S1->Compute(S->Getpresult());
			  vector<arma::fmat*> re1 = S1->Getpresult();
			// re1.at(10)->print("this is presult of S1:\n");
			  //re1.at(10)->print("fill:\n");
              Pro(S1->Getpresult());
              net->aCheck(i);
			  S->clearresult();
			  S1->clearresult();
			// train.print("result cnn:\n");
		
        }
        void Compute(int i){
            S->ComputeF(traininput[i]);          //计算得到12*12 *20矩阵， 存入S presult中， 
		//	traininput.at(i)->print("this is input:\n");
			vector<block*> w = S->Getweight();

			//  w.at(0)->amatrix.print("this is weight of s:\n");
		  //  w.at(10)->amatrix.print("this is weight of s:\n");
			  vector<arma::fmat*> re = S->Getresult();
			//  re.at(0)->print("this is result of S:\n");
			  vector<arma::fmat*> re2 = S->Getpresult();
			//  re2.at(10)->print("this is presult of S:\n");
			  S1->Compute(S->Getpresult());
			  vector<arma::fmat*> re1 = S1->Getpresult();
			// re1.at(10)->print("this is presult of S1:\n");
			  //re1.at(10)->print("fill:\n");
              Pro(S1->Getpresult());
              net->Compute(i);
	
			// train.print("result cnn:\n");
		
        }
		void Back(int i){   
			
		    arma::fmat* backfromenet =  net->Back(i);
			// backfromnet->print();
			  S1->Proback(backfromenet);         //1*800 biancheng 4*4*50;store in bpresult
			//  S1->Getbpresult().at(20)->print("back from net:\n");
			  S1->Backbpresult();         //上采样并得到残差， 8*8*50  存入bresult
			  S1->BackExpresult(16);       // 补0，  16*16*50 存入Expbresult
			  S1->XuanZhuanW();            //将核旋转， 存入XuanZhuanweight
			  vector<arma::fmat*> s1bp = S1->Backpooling(20);
			  S->Setbpresult(s1bp);    // 得到12*12*20  can ca, 存入S bpresult中；
			  

			  vector<arma::fmat*> re2 = S->Getbpresult();
			
			  S1->BackWeight(S->Getpresult());          //跟新权值

	
			  
			  S->Backbpresult();               //24*24*20;
     		  S->XuanZhuanW();
			  vector<arma::fmat*> in;
			  in.push_back(traininput[i]);
			  S->BackWeight(in);
			   //
			  
			  in.clear();
			 /* for(vector<arma::fmat*>::iterator p = s1bp.begin();p!=s1bp.end();p++){
			       delete *p;
			  }*/
			  S->clearresult();
			  S1->clearresult();
			           
		}
		void CTest(int t){
			int size = 10000;
		/*	for (int i = 0; i < size; i++){
				trainlabel.at(i).print();
			}*/
		   for(int i = 0 ;i < t;i++){
			   time_t start = time(NULL);
			   cout << "Number: " << i << '\n';
			   for(int j = 0;j < size;j++){
				 // cout<<"line:"<<j;
			      Test(j);
			   }
			   time_t end = time(NULL);
			   cout << "\n\n\n\ne: " << ep/size << "\ncoast time:\n" << end - start << '\n';
			   cout << "rate" << rate*1.0 / size << '\n';
			   rate = 0;
			   ep = 0;
			  
			   cout<<"trainim size:"<<trainim.size();
			   cout<<"trainlabel size:"<< trainlabel.size();
			   cout<<"traininput size"<< traininput.size();
			  
			  // trainim.clear();
		   }
		}
		void Trian(int t){
			int size = 60000;
		/*	for (int i = 0; i < size; i++){
				trainlabel.at(i).print();
			}*/
		   for(int i = 0 ;i < t;i++){
			   time_t start = time(NULL);
			   cout << "Number: " << i << '\n';
			   for(int j = 0;j < size;j++){
				 // cout<<"line:"<<j;
			      Compute(j);
			      Back(j);
				  
				 
			   }
			   time_t end = time(NULL);
			   cout << "\n\n\n\ne: " << ep/size << "\ncoast time:\n" << end - start << '\n';
			   cout << "rate" << rate*1.0 / size << '\n';
			   rate = 0;
			   ep = 0;
			  
			   cout<<"trainim size:"<<trainim.size();
			   cout<<"trainlabel size:"<< trainlabel.size();
			   cout<<"traininput size"<< traininput.size();
			   Save(i);
			  // trainim.clear();
		   }
		}
         void Pro(vector<arma::fmat*> input){
			 arma::fmat out(1,800);
	         if(input.empty()){cout << "No input"; return;}
			 int si = input.at(0)->n_cols;
	         for(int j = 0; j < input.size();j++){
		        for(int k = 0; k < si;k++){
			         for(int f=0;f < si;f++){
				       out.at(0,j*si*si + k*si + f) = input.at(j)->at(k,f);
		             }
		         }
		     }
			 //out.print();
			 train = out;
			 testt = out;
	     // trainim.push_back(out);
			
	     
	     }



};

