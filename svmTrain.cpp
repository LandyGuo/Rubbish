#include<string>
#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include <cv.h>
#include <highgui.h>
#include <opencv.hpp>
#include "extractSiftFeatures.h"
using namespace std;
using namespace cv;




string result_save_path("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\cookeddata\\Results\\res.txt");


#define CLASSES 4
//*feature-dims,landmarks(42)*128(Sift-feature descriptor)*/
#define SIFT_KEY_POINTS 40
#define SIFT_DIMS 128
#define SIFT_FEATURE_DIMS SIFT_KEY_POINTS*SIFT_DIMS

//*sample stucture*/
typedef struct TrainingSample
{
	vector<int> feature;//feature vector
	int label;//sample label
}TrainingSample;

//*class definition*/
class TrainingSvm
{
	public:
		string indexFilePath;
		string SVM_SavePath;
		vector<TrainingSample> SampleVector;
private:
	int imageCount;
	public:
		TrainingSvm(string indexFilePath,string SVM_SavePath):indexFilePath(indexFilePath),SVM_SavePath(SVM_SavePath){createTrainingData();};
		vector<string> loadIndexFile();
		static string get_One_ImagePath(string imgPath);
		static string convertInt2String(int const&i);
		vector<int> get_One_featureVector(string image_path);
		void  createTrainingData();
		vector<int> getTrainingFeatureIndex();//choose data to trainSVM
		void BeginTraining();//start training
		void modelValidation();//validate the SVM model,calculate precision
		void outputResults(int  ResultMatrix[][CLASSES]);

	private:
		void saveSVM();//where to save SVM-model
};
//function definition
//************************************************************************/
//* utils:convert int to string                                                                     */
//************************************************************************/
string TrainingSvm::convertInt2String(int const&i)
{
	stringstream ss;
	string i_str;
	ss<<i;
	ss>>i_str;
	return i_str;
}
//************************************************************************/
//*load index file,each row as a string                                    */
//************************************************************************/
vector<string> TrainingSvm::loadIndexFile()
{
	ifstream fs(this->indexFilePath);
	vector<string> result;
	if (fs.fail())
	{
		cout<<"ERROR: can not open file!"<<endl;
		exit(-1);
	}
	string line;//line-format:S005 001 3 3
	//int lineNum = 0;
	while (getline(fs,line))
	{
		//if (lineNum>=30)
		//{
		//	break;
		//}
		result.push_back(line);
		//lineNum++;
	}
	return result;
}
//************************************************************************/
//* input: S005 001 2 
//  output: S005_001_EAI2.png  //imagePath                                       */
//************************************************************************/
string TrainingSvm::get_One_ImagePath(string imgPath)
{
	return "C:\\Users\\guoqingpei\\Desktop\\newjingtai\\cookeddata\\" + imgPath;
}
//************************************************************************/
//* featureExtractor:extract feature vector from one image               */
//************************************************************************/
vector<int> TrainingSvm::get_One_featureVector(string image_path)
{
	Mat image =imread(image_path, 0);//读取图像

	vector<KeyPoint> keypoints;
	Mat descriptors;
	vector<int> vec;
	SIFT sift = SIFT(SIFT_KEY_POINTS);
	sift(image, Mat(), keypoints, descriptors, false);
	//cout << endl << descriptors << endl;

	for (int i = 0; i < descriptors.rows; i++)
	{
		for (int j = 0; j < 128; j++)
		{
			int val = (int)descriptors.at<float>(i, j);
			vec.push_back(val);
			//cout << val << "\t";
		}
	}
	//feature 补0
	while (vec.size() < SIFT_FEATURE_DIMS)
	{
		vec.push_back(0);
	}
	return vec;
}
//************************************************************************/
//*get all image features and collect feature and labels to build training sample */
//************************************************************************/
void TrainingSvm::createTrainingData()
{
	vector<string> indexFile = loadIndexFile();
	this->imageCount = indexFile.size();
	for(auto p = indexFile.cbegin();p!=indexFile.cend();p++)
	{
		istringstream isstream(*p);
		string imgpath;
		int label;
		isstream >> imgpath >> label;

		string imgPath = this->get_One_ImagePath(imgpath);
		vector<int> feature = this->get_One_featureVector(imgPath);
		cout<<"load Image:"<<imgPath<<endl
			<<"extracting feature complete"<<endl;
		TrainingSample ts;
		ts.feature = feature;
		ts.label = label;
		this->SampleVector.push_back(ts);
		
	}
}
//************************************************************************/
//* choose index in vector<TrainingSample> as training data,the left are test data set*/
//************************************************************************/
vector<int> TrainingSvm::getTrainingFeatureIndex()
{
	vector<int> trainingIndex;
	for (auto i=0;i<this->SampleVector.size();i++)
	{
		trainingIndex.push_back(i);
	}
	random_shuffle(trainingIndex.begin(),trainingIndex.end());
	return trainingIndex;
}
//************************************************************************/
//* start training                                                    */
//************************************************************************/
void TrainingSvm::BeginTraining()
{
	//choose 100% to train
	vector<int> trainingIndex = this->getTrainingFeatureIndex();
	int sampleNum = trainingIndex.size();
	vector<int> labelsVector;
	Mat trainingDataMat(sampleNum, SIFT_FEATURE_DIMS, CV_32F, cv::Scalar::all(0.0));
	for (auto i = 0; i<sampleNum; i++)
	{
		TrainingSample ts = this->SampleVector[i];
		for (auto j = 0; j<SIFT_FEATURE_DIMS; j++)
		{
			trainingDataMat.at<float>(i, j) += ts.feature[j];
		}
		labelsVector.push_back(ts.label);
	}
	//Attention:SVM labels must continuous 0-7,if 0-5,7 then error will occur
	Mat trainingLabelsMat = Mat(labelsVector);
	/*--------------start training SVM-------------*/
	CvSVM SVM;
	/*-------------------SVM params setting---------*/
	CvSVMParams params;
	params.kernel_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.C = 1;    
	params.p = 5e-3;   
	params.gamma = 0.01;  
	params.term_crit = cvTermCriteria(CV_TERMCRIT_EPS, 100, 5e-3);
		
	CvParamGrid nuGrid = CvParamGrid(1, 1, 0.0);
	CvParamGrid coeffGrid = CvParamGrid(1, 1, 0.0);
	CvParamGrid degreeGrid = CvParamGrid(1, 1, 0.0);

	cout<<"-------------Begin Training SVM------------------"<<endl;
	SVM.train_auto(trainingDataMat, trainingLabelsMat, cv::Mat(), cv::Mat(), params,
		10,
		CvSVM::get_default_grid(CvSVM::C),
		CvSVM::get_default_grid(CvSVM::GAMMA),
		CvSVM::get_default_grid(CvSVM::P),
		nuGrid,
		coeffGrid,
		degreeGrid, false);
	/*-------------save SVM model---------------*/
	cout<<"saving SVM,path: "+this->SVM_SavePath<<endl;
	SVM.save(SVM_SavePath.c_str());
}

/************************************************************************/
/* calculate precison rate                                              */
/************************************************************************/
void TrainingSvm::modelValidation()
{
	CvSVM SVM;
	SVM.load(SVM_SavePath.c_str());//loadSVM

	int totalSample = 0, rightSample = 0;
	int ResultMatrix[CLASSES][CLASSES] = { 0 };
	//get test data
	for(int i= 0;i<this->imageCount;i++)
	{
		TrainingSample ts = this->SampleVector[i];
		cv::Mat sampleMat(1, SIFT_FEATURE_DIMS, CV_32F, cv::Scalar::all(0.0));
		for (int j = 0; j < SIFT_FEATURE_DIMS; j++)
		{
			sampleMat.at<float>(j) = ts.feature[j];
		}
		int predictLabel = SVM.predict(sampleMat);
		int realLabel = ts.label;
		ResultMatrix[ts.label][predictLabel]++;
	}
	//output precision
	outputResults(ResultMatrix);
}

/************************************************************************/
/* output Results                                                                     */
/************************************************************************/
void TrainingSvm::outputResults(int  ResultMatrix[][CLASSES])
{
	int total = 0, right = 0;
	for (int i = 0; i < CLASSES; i++)
	{
		for (int j = 0; j < CLASSES; j++)
		{
			total += ResultMatrix[i][j];
		}
		right += ResultMatrix[i][i];
	}
	//output precision to text-file
	ofstream outfile(result_save_path, ofstream::out | ofstream::app);
	string trainingType = "1v8";
	trainingType += "\t 1V1";
	outfile << "----------------------------------------------------------------------" << endl;
	outfile << trainingType << endl
		<< "total:" << total << "\t"
		<< "right:" << right << endl;
	string classes[CLASSES] = { "qt", "qh", "qk", "xh" };
	//0= neutral, 1 = anger, 2 = contempt, 3 = disgust, 4 = fear, 5 = happy, 6 = sadness, 7 = surprise
	outfile << "\t";
	for (int i = 0; i < CLASSES; i++)
		outfile << "\t" << classes[i] << "\t";
	outfile << "\t" << "precision" << endl;
	for (int i = 0; i < CLASSES; i++)
	{
		outfile << classes[i] << "\t";
		int sum = 0;
		for (int j = 0; j < CLASSES; j++)
		{
			outfile << "\t" << ResultMatrix[i][j] << "\t";
			sum += ResultMatrix[i][j];
		}
		outfile << "\t" << ResultMatrix[i][i] / float(sum) << endl;
	}
}



int main()
{
	TrainingSvm * ptrain = new TrainingSvm(string("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\cookeddata\\rp\\va.txt"),
		string("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\cookeddata\\SVMs\\Sift_SVM_1v8.xml"));
	//ptrain->BeginTraining();
	ptrain->modelValidation();
}
