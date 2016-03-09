#include "layer.h"
#include <fstream>


void swapBuffer(char* buf)     //used in read data head;
{
	char temp;
	temp = *(buf);
	*buf = *(buf + 3);
	*(buf + 3) = temp;

	temp = *(buf + 1);
	*(buf + 1) = *(buf + 2);
	*(buf + 2) = temp;
}

class Net{
private:
	int firstnerous;       //this = 60000
	int col;         //this = 361
	int h_hide;     // this = 3;
	int error;       //count the error number of size(6000)
	int rate;        //count the right number of size
	vector<Layer> e_hide;   //store the layer of net;e_hide[0] is start layer, only has input,  and e_hide[h_hide] is output layer;

public:
	Net(){
	}
	Net(const string filename, const string labelname, int hide,int col,int firstnerous){
		ReadDate();
		ReadTest();
		date();
		int subnerous = (firstnerous - CLASS) / hide;
		rate = 0;
		this->firstnerous = firstnerous;
		h_hide = hide;
		Layer* s = new Layer(0, col);
		e_hide.push_back(*s);

		for (int i = 1; i < h_hide; i++){
			Layer* p = new Layer(e_hide.at(i - 1).GetSize(), firstnerous);
			firstnerous -= subnerous;
			e_hide.push_back(*p);
		}
		Layer *re = new Layer(e_hide.at(hide - 1).GetSize(), CLASS);
		e_hide.push_back(*re);
	}
	Net(const string filename, const string labelname,int col){
		ReadDate();
		ReadTest();
		date();
		Layer* s = new Layer(0, col);
		e_hide.push_back(*s);
		Load();
	}
	int Compute(int i);      //
	bool ReadDate();
	bool ReadTest();
	void aCheck(int t);
	void Back(int i);
	void CTest();
	void Train(int a);
	bool Save(int t);
	void date();
	bool Load();
	bool Save(const string filename);
	bool ReadDate1();
};
void GetROI(Mat& src, Mat& dst)
{
	int left, right, top, bottom;
	left = src.cols;
	right = 0;
	top = src.rows;
	bottom = 0;

	//Get valid area
	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			if (src.at<uchar>(i, j) > 0)
			{
				if (j<left) left = j;
				if (j>right) right = j;
				if (i<top) top = i;
				if (i>bottom) bottom = i;
			}
		}
	}
	int width = right - left;
	int height = bottom - top;
	int len = (width < height) ? height : width;

	//Create a squre
	dst = Mat::zeros(len, len, CV_8UC1);

	//Copy valid data to squre center
	Rect dstRect((len - width) / 2, (len - height) / 2, width, height);
	Rect srcRect(left, top, width, height);
	Mat dstROI = dst(dstRect);
	Mat srcROI = src(srcRect);
	srcROI.copyTo(dstROI);
}

bool Net::ReadDate(){
	const char fileName[] = "train-images.idx3-ubyte";
	const char labelFileName[] = "train-labels.idx1-ubyte";


	ifstream ifs(fileName, ios_base::binary);
	ifstream lab_ifs(labelFileName, ios_base::binary);

	if (ifs.fail() == true)
		return false;
	if (lab_ifs.fail() == true)
		return false;
	
	char magicNum[4], ccount[4], crows[4], ccols[4];
	ifs.read(magicNum, sizeof(magicNum));   //read head of file
	ifs.read(ccount, sizeof(ccount));
	ifs.read(crows, sizeof(crows));
	ifs.read(ccols, sizeof(ccols));

	int count, rows, cols;
	swapBuffer(ccount);
	swapBuffer(crows);
	swapBuffer(ccols);

	memcpy(&count, ccount, sizeof(count));
	memcpy(&rows, crows, sizeof(rows));
	memcpy(&cols, ccols, sizeof(cols));

	lab_ifs.read(magicNum, sizeof(magicNum));
	lab_ifs.read(ccount, sizeof(ccount));

	Mat src = Mat::zeros(rows, cols, CV_8UC1);
	Mat dst;
	int total = 0;
	char label = 0;
	char p = 0;
	IMG img;
	while (!ifs.eof()){
		if (total >= count)
			break;
		total++;
		lab_ifs.read(&label, 1);
		img.l_label = label;
		ifs.read((char*)src.data, rows*cols);   
		for (int i = 0; i < src.cols; i++){
			for (int j = 0; j<src.rows; j++){
				img.image[i*src.cols + j] = src.at<uchar>(i, j) / 255.0;
			}
			
		}

		im.push_back(img);

	}
	dst.~Mat();
	src.~Mat();

	return true;
}

// pack source date into batching.
void Net::date(){
 	for(int i = 0; i < im.size();){
		arma::fmat p(BATCHING,28*28);
		arma::fmat q(BATCHING,1);

	    for(int j = 0; j < BATCHING;j++){
		     for(int f = 0; f< 28*28;f++){
				p.at(j,f) = (float)im.at(i+j).image[f];
				q.at(j,0)=(float)im.at(i+j).l_label;
			 }
		}
		trainim.push_back(p);
		trainlabel.push_back(q);
	    i = i+BATCHING;
	}

	for (int i = 0; i < test.size();){
		arma::fmat p(BATCHING, 28 * 28);
		arma::fmat q(BATCHING, 1);
		for (int j = 0; j < BATCHING; j++){
			for (int f = 0; f< 28 * 28; f++){
				p.at(j, f) = (float)im.at(i + j).image[f];
				q.at(j, 0) = (float)im.at(i + j).l_label;
			}
		}
		testim.push_back(p);
		testlabel.push_back(q);
		i = i + BATCHING;
	}
}


bool Net::ReadTest(){
    const char fileName[] = "t10k-images.idx3-ubyte";
	const char labelFileName[] = "t10k-labels.idx1-ubyte";

	ifstream ifs(fileName, ios_base::binary);
	ifstream lab_ifs(labelFileName, ios_base::binary);
	if (ifs.fail() == true)
		return false;
	if (lab_ifs.fail() == true)
		return false;
	
	char magicNum[4], ccount[4], crows[4], ccols[4];
	ifs.read(magicNum, sizeof(magicNum));   //read head of file
	ifs.read(ccount, sizeof(ccount));
	ifs.read(crows, sizeof(crows));
	ifs.read(ccols, sizeof(ccols));

	int count, rows, cols;
	swapBuffer(ccount);
	swapBuffer(crows);
	swapBuffer(ccols);

	memcpy(&count, ccount, sizeof(count));
	memcpy(&rows, crows, sizeof(rows));
	memcpy(&cols, ccols, sizeof(cols));

	lab_ifs.read(magicNum, sizeof(magicNum));
	lab_ifs.read(ccount, sizeof(ccount));

	Mat src = Mat::zeros(rows, cols, CV_8UC1);
	Mat dst;
	int total = 0;
	char label = 0;
	char p = 0;
	IMG img;
	while (!ifs.eof()){
		if (total >= count)
			break;
		total++;
		lab_ifs.read(&label, 1);
		img.l_label = label;

		ifs.read((char*)src.data, rows*cols);
		

		for (int i = 0; i < src.cols; i++){
			for (int j = 0; j<src.rows; j++){
				img.image[i*src.cols + j] =src.at<uchar>(i, j) / 255.0;
			}
			
		}

		test.push_back(img);
	}
	dst.~Mat();
	src.~Mat();
}


int Net::Compute(int t){             //
	if (!e_hide.empty())
			e_hide.at(0).asetStart(t);
	for (int i = 1; i <= h_hide; i++){
		e_hide.at(i).Compute1(e_hide.at(i - 1).agetoutput());
	}
	arma::fmat *out = e_hide.at(h_hide).agetoutput();  //out is 60*10 ,each col is a rate of number(0-9) 
	arma::umat outt;

	outt.set_size(BATCHING,1);
	for(int i = 0;i<BATCHING;i++){
        int p = -1;
		double x = 0;
		for(int j = 0; j < CLASS;j++){
			if((*out).at(i,j) > x){
			    x = (*out).at(i,j);
				p = j;
			}
		}
		outt.at(i,0) = p;
		//cout<<"pre: "<<p<<" result:"<<trainlabel[t]<<endl;
		
	}
	arma::umat tr = (trainlabel[t]==outt);

	int resu = accu(tr);

	rate+=resu;
	return 1;
}

void Net::Back(int t){
	int size = e_hide.size();
	if (size - 2 < 0){
		cout << "the number of layer error \n";
		return;
	} 
	e_hide.at(e_hide.size() - 1).aUpdateWofend(trainlabel[t], *e_hide.at(size - 2).agetoutput());
	for (int i = size - 2; i >= 1; i--){
		e_hide.at(i).aUpdateW(*e_hide.at(i - 1).agetoutput(), *e_hide.at(i + 1).agetback());
	}
}

void Net::aCheck(int t){
	if (!e_hide.empty())
		e_hide.at(0).asetTStart(t);
	for (int i = 1; i <= h_hide; i++){
		e_hide.at(i).Compute1(e_hide.at(i - 1).agetoutput());
	}
	arma::fmat *out = e_hide.at(h_hide).agetoutput(); 
	arma::umat outt;
	outt.set_size(BATCHING, 1);
	for (int i = 0; i<BATCHING; i++){
		int p = -1;
		double x = 0;
		for (int j = 0; j < CLASS; j++){
			if ((*out).at(i, j) > x){
				x = (*out).at(i, j);
				p = j;
			}
		}
		outt.at(i, 0) = p;
		cout<<"pre:"<<p<<" True:"<<testlabel[t]<<endl;
	}
	//cout<<"batching: "<<t<<endl;
	//testlabel[t].print();
	//outt.print();
	//cout<<endl;
	
	arma::umat tr = (testlabel[t] == outt);
	int resu = accu(tr);

	rate += resu;

}

bool Net::Save(int t){
	char p[6];
	itoa(t,p,10);
	string pp(p);
	string name = "save"+pp+".txt";
	ofstream ou(name);
	ou << h_hide << '\n';
	ou<<firstnerous<<'\n';
	for (int i = 0; i < h_hide+1; i++){
		e_hide.at(i).Save(ou);
	}
	ou.close();
	return true;
}
bool Net::Load(){
    char p[6];
	itoa(1,p,10);
	string pp(p);
	string name = "save"+pp+".txt";
	ifstream ou(name);
	ou >>h_hide;
	ou>>firstnerous;
	for(int i = 1;i < h_hide+1;i++){
		Layer *p= new Layer();
		e_hide.push_back(*p);
	}
	
	for (int i = 0; i < h_hide+1; i++){
		e_hide.at(i).Load(ou);
	}
	ou.close();
	return true;
}
void Net::Train(int Ti){
	int times = 0;
	 double r = 0;
	while (times < Ti){
	
		time_t start = time(NULL);
		ep = 0;
		for (int i = 0; i < im.size()/BATCHING; i++){
		//	cout<<"line: "<<i<<endl;
			Compute(i);      
			Back(i);         
		}
		time_t end=time(NULL);
		cout<<"coast time:"<<end-start<<endl;
		cout << ep / im.size() << ' ';
		r=rate*1.0 / im.size() ;
		cout << "True:" << r << '\n';
		
		rate = 0;
		times++;
	    Sleep(1000);
		cout<<"times: "<<times<<'\n';

		CTest();
	//	if(times%SAVETIME == 0){
			Save(times);
			cout<<"save: "<<times<<'\n';
	//	}
		rate = 0;
	}
}

void Net::CTest(){
	int times = 0;
	int k = 0;
	rate = 0;
    while(k<testim.size()){
		aCheck(k);
		k++;
	}
	cout << "test true: ";
	cout<<rate*1.0/test.size()<<'\n';

}







