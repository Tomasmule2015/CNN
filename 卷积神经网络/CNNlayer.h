#include"head.h"

class CLayer{
    private:
    	int number;
    	int weightsize;
    	int resultsize;           //biru 24*24
    	int presultsize;          // 12*12
		arma::fmat theate;
	    vector<block*> weight;
    	vector<block*> bweight;
    	vector<arma::fmat*> result;        //卷积层结果
    	vector<arma::fmat*> bresult;   //卷积层残差
    	vector<arma::fmat*> presult;   //pooling层结果
    	vector<arma::fmat*> bpresult;   //pooling层残差
		vector<arma::fmat*> Expbresult;  // 卷积层残差补0
		vector<arma::fmat*> XuanZhuanweight; //经过旋转的权值
		
    public:
        CLayer(int numofbefore,int num,int wsize,int resize){
        	number = num;
        	weightsize = wsize;
        	resultsize = resize;
        	presultsize = resultsize/2;
			
			
			for(int i = 0; i < number*numofbefore;i++){
        		block* p1 = new block(weightsize);
        		weight.push_back(p1);
        	}

        }
		~CLayer(){
			for(vector<block*>::iterator s = weight.begin();s!=weight.end();s++){
			       delete *s;
			}
			weight.clear();
		}

		void Setbpresult(vector<arma::fmat*> s){
			bpresult = s;}
    	vector<block*> Getweight(){return weight;}
    	vector<block*> Getbweight(){return bweight;}
    	vector<arma::fmat*> Getresult(){return result;}
    	vector<arma::fmat*> Getbresult(){return bresult;}   //can cha
    	vector<arma::fmat*> Getpresult(){return presult;}
    	vector<arma::fmat*> Getbpresult(){return bpresult;}
		vector<arma::fmat*> GetExpbresult(){return Expbresult;}
        
        arma::fmat* ComputeF(arma::fmat* input);
    	arma::fmat* ComputeCNN(arma::fmat* front,block* weight);
		arma::fmat* ComputeC(arma::fmat* front,arma::fmat* back);
	    arma::fmat* Expand(arma::fmat* input,int size);
        arma::fmat* UpSample(arma::fmat* input);
        arma::fmat* Pooling(arma::fmat* input);
        arma::fmat* XuanZhuan(arma::fmat input);
		void Compute(vector<arma::fmat*> input);
		void Save(int i){

			char ti[10];
			_itoa_s(i, ti, 10);
			string times(ti);

			char p[10];
		    _itoa_s(number,p,10);
		    string qq(p);
		    for(int i = 0; i < weight.size();i++){
                   char q[10];
				   _itoa_s(i, q, 10);
                   string pp(q);
                   weight.at(i)->amatrix.save(times+qq+pp);
		    }
		}
		void Load(int size,int i){
			char ti[10];
			_itoa_s(i, ti, 10);
			string times(ti);
			char p[10];
			_itoa_s(number, p, 10);
		    string qq(p);
		    for(int i = 0; i < size;i++){
                   char q[10];
				   _itoa_s(i, q, 10);
                   string pp(q);
                   weight.at(i)->amatrix.load(times+qq+pp);
		    }
		}
		void Backbpresult(){                 //上采样
			arma::fmat ss(result.at(0)->n_cols,result.at(0)->n_cols);
			ss.ones();
			for(int i = 0; i < bpresult.size();i++){
				arma::fmat* p = UpSample(bpresult.at(i));
				arma::fmat* q = new arma::fmat(p->n_cols,p->n_rows);
				//p->print();
				//result.at(i)->print();
			    *q = *p % *result.at(i)%(ss - *result.at(i));
				//q->print();
				delete p;
				bresult.push_back(q);  
			}
//				bpresult.at(0)->print();
	//			bresult.at(0)->print();
		}
		void BackWeight(vector<arma::fmat*> input){       //跟新权值
			int si = 5;
			int siz = bresult.at(0)->n_cols; 
			 if(input.empty()){return;} 
			  //input.at(0)->print();bresult.at(0)->print();
			   for(int i = 0; i < number;i++){
				   double p2 = arma::accu(*bresult.at(i));
						   for(int t = 0; t <input.size();t++){
							  
							   arma::fmat * p1 = (ComputeC(input.at(t),bresult.at(i)));
							 
							   block* p3 = new block();
							   p3->Set(*p1);
							   delete p1;
							  // p3->theate = p2;
							   bweight.push_back(p3);
						   }
			} 
			//   bweight.at(0)->amatrix.print();
			//   weight.at(0)->amatrix.print();
			   for(int i = 0; i < weight.size();i++){
				  // bweight.at(i)->amatrix.print();
			     //  weight.at(i)->amatrix.print();
				   weight.at(i)->amatrix -= 0.05*bweight.at(i)->amatrix;
				 //  weight.at(i)->theate -= 0.5*bweight.at(i)->theate;
				 //  weight.at(i)->amatrix.print();
			   }
			//   weight.at(0)->amatrix.print();

		}
		void BackExpresult(int t){           //补0
			for(int i = 0; i< bresult.size();i++){
				Expbresult.push_back(Expand(bresult.at(i),t));
				//bresult.at(i)->print("\n");
			}
			//bresult.at(0)->print();
			//Expbresult.at(0)->print();
		}
		void XuanZhuanW(){                    //旋转weight
			for(int i = 0; i < weight.size();i++){
			
				XuanZhuanweight.push_back(XuanZhuan(weight.at(i)->amatrix));
			}
	//		weight.at(0)->amatrix.print();
		//	XuanZhuanweight.at(0)->print();
		}
		vector<arma::fmat*> Backpooling(int bsize){
		     vector<arma::fmat*> back;
			 int si = Expbresult.at(0)->n_cols -4;
			 for(int i = 0 ; i < bsize;i++){
			    arma::fmat* p1 = new arma::fmat(si,si);
				p1->zeros();
			//	p1->print();
				for(int j = 0; j < number;j++){
					arma::fmat* p = ComputeC(Expbresult.at(j), XuanZhuanweight.at(j*bsize + i));
					*p1 += *p;
					delete p;
				}
				//p1->print();
				back.push_back(p1);
			 }
			 return back;
		}
			 
    	void clearresult(){
			for(vector<arma::fmat*>::iterator iter = result.begin(); iter != result.end();iter++){
				delete *iter;
			}
			for (vector<arma::fmat*>::iterator iter = bresult.begin(); iter != bresult.end(); iter++){
				delete *iter;
			}
			for (vector<arma::fmat*>::iterator iter = presult.begin(); iter != presult.end(); iter++){
				delete *iter;
			}
			for (vector<arma::fmat*>::iterator iter = bpresult.begin(); iter != bpresult.end(); iter++){
				delete *iter;
			}
			for (vector<block *>::iterator iter = bweight.begin(); iter != bweight.end(); iter++){
				delete *iter;
			}
			for (vector<arma::fmat*>::iterator iter = Expbresult.begin(); iter != Expbresult.end(); iter++){
				delete *iter;
			}
			for (vector<arma::fmat*>::iterator iter = XuanZhuanweight.begin(); iter != XuanZhuanweight.end(); iter++){
				delete *iter;
			}
    		result.clear();
    		bresult.clear();
    		presult.clear();
    		bpresult.clear();
			bweight.clear();
			Expbresult.clear();
			XuanZhuanweight.clear();
    	}
		void  Proback(arma::fmat* back){            //cun ru bpresultzhong 
	       
        	for(int i = 0; i < (*back).n_rows;i++){    //dui mei yi han
		        for(int j = 0; j < (*back).n_cols;j+=16){
		        arma::fmat* s = new arma::fmat;

			   *s = back->submat(i,j,arma::size(1,16));
			   s->set_size(4,4);
			   *s = s->t();
			  
			   bpresult.push_back(s);
		       }
	        }
	     // bpresult.at(1)->print();
        }
};
void CLayer::Compute(vector<arma::fmat*> input){
    if(input.empty()){return;}
	
	//input.at(0)->submat(2,2,arma::size(5,5)).print();
	//weight.at(0)->amatrix.print();
	int sssss = input.at(0)->n_cols;
	int si = sssss -4;
	int siz =5;
	int dmatsize = number;
	for(int i = 0; i < number;i++){
		arma::fmat *re = new arma::fmat(si,si);
		re->zeros();
		
		       for(int t = 0; t < input.size();t++){
				  arma::fmat * p = ComputeCNN(input.at(t),weight.at(i*input.size() + t));
				  *re += *p;
				  delete p;
     		   }
	     	
		//float s = weight.at(i)->theate;
		//arma::fmat theat(si/N,si/N);
		//theat.fill(s);
		//*re = *re + theat;
		//re->print("\n");
		
	//	re->print("\n");
		re->transform([](float val){return 1 / (1 + exp(-val)); });
		result.push_back(re);

		arma::fmat *res = new arma::fmat(si / N, si / N);
		arma::fmat *s = Pooling(re);
		*res = *s;
		delete s;
		//*res = *res + theat;
		//res->transform([](float val){return 1 / (1 + exp(-val)); });
		presult.push_back(res);
	}


}
arma::fmat* CLayer::ComputeF(arma::fmat* input){
	if(input == NULL){return NULL;}
	
	int si = input->n_cols -weightsize +1;
	for(int i = 0 ; i < number;i++){
	//	float s = weight.at(i)->theate;
		//input->submat(10,10,arma::size(5,5)).print();
		//weight.at(i)->amatrix.print();

        arma::fmat* p = ComputeCNN(input,weight.at(i));
		//cout<<p->at(10,10)<<'\n';
		//p->print("this is result of CNN:\n");
	  // 	arma::fmat theat(si/N,si/N);
		//theat.fill(s);
		//p->print();
		p->transform([](float val){return 1/(1+exp(-val));});
		result.push_back(p);
		//cout<<p->at(10,10)<<'\n';
		
		arma::fmat *s = Pooling(p);
		
		//for(int i = 0;i< si/N;i++){          
		//    for(int j = 0; j< si/N;j++){
		//		//p->submat(i*N, j*N, arma::size(N, N)).print();
		//		float ppp = arma::accu(p->submat(i*N,j*N,arma::size(N,N)))/4.0;
		//		pre->at(i,j) = ppp;
		//	}
		//}
	//	*pre = *pre + theat;
		//pre->print("this is presult of CNN:\n");
		//pre->transform([](float val){return 1 / (1 + exp(-val)); });
		//pre->print("this is presult of CNN2:\n");
		presult.push_back(s);
	}
	return NULL;
}
arma::fmat* CLayer::Pooling(arma::fmat* input){
	int si = input->n_cols;
	arma::fmat *result = new arma::fmat(si/N,si/N);
	for(int i = 0;i< si/N;i++){          
		    for(int j = 0; j< si/N;j++){
				float ppp = arma::accu(input->submat(i*N,j*N,arma::size(N,N))/(N*N));
				 result->at(i,j) = ppp;
			}
	}
	return result;
}
arma::fmat* CLayer::ComputeC(arma::fmat* front,arma::fmat* w){
         int size1 = front->n_cols;
    int size2 = w->n_cols;
    int si = size1 - size2 +1;
    arma::fmat* re = new arma::fmat(si,si);
    for(int i = 0; i < si;i++){
		for(int j = 0; j < si;j++){
			float jjj = arma::accu((front->submat(i,j,arma::size(size2,size2)))%(*w));
			re->at(i,j) = jjj;
		}
	}
     return re;
}
arma::fmat* CLayer::ComputeCNN(arma::fmat* front,block* w){
    int size1 = front->n_cols;
    int size2 = w->amatrix.n_cols;
    int si = size1 - size2 +1;
    arma::fmat* re = new arma::fmat(si,si);
    for(int i = 0; i < si;i++){
		for(int j = 0; j < si;j++){
			float jjj = arma::accu((front->submat(i,j,arma::size(size2,size2)))%(w->amatrix));
			re->at(i,j) = jjj;
		}
	}
     return re;
}
arma::fmat* CLayer::XuanZhuan(arma::fmat input){
//	input->print();
	 arma::fmat* A = new arma::fmat();
	 A->set_size(input.n_rows,input.n_cols);
	 for(int i = 0;i < input.n_rows;i++){
		 for(int j = 0; j < input.n_cols;j++){
		     A->at(i,j) = input.at(i,j);
		 }
	 }
	 double p = 0;
	 p =  A->n_rows/2;
	 if(A->n_rows%2 == 0){ 
	   for(int i = 0; i < A->n_cols;i++){
		 for(int j  = 0; j < p;j++){
		    float p = A->at(i,j);
			A->at(i,j) = A->at(A->n_cols-1-i,A->n_rows-1-j);
			A->at(A->n_cols-1-i,A->n_rows-1-j) = p;
		 }
	   }
	 }else{
		for(int i = 0; i < (A->n_cols)/2;i++){
			for (int j = 0; j < (A->n_cols); j++){
				float p = A->at(i, j);
				A->at(i, j) = A->at(A->n_cols - 1 - i, A->n_rows - 1 - j);
				A->at(A->n_cols - 1 - i, A->n_rows - 1 - j) = p;
		    }
	     }
		int mid = (A->n_cols+1)/2;
		for(int j = 0; j < A->n_cols/2;j++){
			float p = A->at(mid-1,j);
			//cout<<p<<endl;
			A->at(mid-1,j) = A->at(mid-1,A->n_cols -1-j);
			//cout<<A->at(mid-1,j);
			A->at(mid-1,A->n_cols-1-j) = p;
		}
    }
//	 A->print();
		return A;
}
arma::fmat* CLayer::UpSample(arma::fmat* b){
    int si = 2*b->n_cols;
	arma::fmat* p = new arma::fmat(si,si);
//	b->print();
	p->zeros();
	for(int i = 0; i < b->n_cols; i++){
		for(int j = 0; j < b->n_cols; j++){
		   p->submat(i*2,j*2,arma::size(2,2)).fill(b->at(i,j)/4.0);
	   }
	}
//	p->print();
	return p;
}

arma::fmat* CLayer::Expand(arma::fmat* input, int si){   //add '0' 
	arma::fmat* p = new arma::fmat(si,si);
	p->zeros();

	int s =(si - input->n_cols)/2;
	for(int i = 0; i < input->n_cols;i++){
	    for(int j = 0; j < input->n_cols;j++){
			p->at(s+i,s+j) = input->at(i,j);
		}
	}

	return p;
}