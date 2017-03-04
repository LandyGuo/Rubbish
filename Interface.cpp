#include<iostream>
#include<string>
#include<algorithm>
#include <cv.h>
#include <highgui.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv.hpp>
#include <assert.h>

using namespace std;
using namespace cv;


#define STATES 4
#define TRAIN_IMAGE_WIDTH 192  
#define TRAIN_IMAGE_HEIGHT 64
#define SIFT_KEY_POINTS 40
#define SIFT_DESCRIPTOR_DIMS 128
#define SIFT_FEATURE_DIMS SIFT_KEY_POINTS*SIFT_DESCRIPTOR_DIMS

/*Recognize Result*/
//{'bk': 0, 'qh' : 1, 'qk' : 2, 'xh' : 3}
typedef enum SwitchState
{
	STATE_HALF_OPEN,
	STATE_FULLY_CLOSED,
	STATE_FULLY_OPEN,
	STATE_NEARLY_CLOSED,
}SwitchState;


/*feature extractor*/
class FeatureExtractor
{
public:
	virtual vector<int> extract(Mat const & image) const = 0;
};

class SiftFeatureExtractor : public FeatureExtractor
{
private:
	SIFT sift;
	unsigned int expectKeyPoints;
public:
	SiftFeatureExtractor(unsigned int expectKeyPoints) :
		sift(expectKeyPoints),
		expectKeyPoints(expectKeyPoints){};

	vector<int> extract(Mat const & image) const override
	{
		vector<KeyPoint> keypoints;
		Mat descriptors;
		vector<int> vec;
		sift(image, Mat(), keypoints, descriptors, false);
		for (int i = 0; i < descriptors.rows; i++)
		{
			for (int j = 0; j < SIFT_DESCRIPTOR_DIMS; j++)
			{
				int val = (int)descriptors.at<float>(i, j);
				vec.push_back(val);
				//cout << val << "\t";
			}
		}
		while (vec.size() < SIFT_FEATURE_DIMS)
			vec.push_back(0);
		return vec;
	}
};


/*Recognize State*/
class Predicator
{
public:
	virtual SwitchState predict(Mat const &feature) const = 0;
	virtual SwitchState predict(vector<int> const & feature) const = 0;
};

class SvmPredicator : public Predicator
{
private:
	string svmPath;//svmPath: path to save trained SVMs
	int featureDim;//Size: feature size for SVM predict 
	CvSVM SVM;
public:
	SvmPredicator(string svmPath, int featureDim) :svmPath(svmPath), featureDim(featureDim){
		SVM.load(svmPath.c_str());
	}//loadSVM

	SwitchState predict(Mat const & feature) const override
	{
		assert(feature.rows == 1 && feature.cols == featureDim);
		//SVM predict
		int predictLabel = SVM.predict(feature);
		return SwitchState(predictLabel);
	}

	SwitchState predict(vector<int> const & feature)const override
	{
		Mat sampleMat(1, SIFT_FEATURE_DIMS, CV_32F, cv::Scalar::all(0.0));
		for (int j = 0; j < SIFT_FEATURE_DIMS; j++)
		{
			sampleMat.at<float>(j) = feature[j];
		}
		return this->predict(sampleMat);
	}
};


class MultiStateRecognizer
{
private:
	Predicator const & predicator;
	FeatureExtractor const & extractor;// Patch Size for recognize
public:
	MultiStateRecognizer(Predicator & predicator, FeatureExtractor& extractor)
		:predicator(predicator), extractor(extractor){}

	SwitchState recognize(Mat const & imgPatch) const//recognize one patch
	{
		//change to gray if needed
		Mat grayMat;
		if (imgPatch.channels() != 1)
			cvtColor(imgPatch, grayMat, CV_BGR2GRAY);
		else
			grayMat = imgPatch;
		//imshow("test", grayMat);
		//waitKey();
		//resize current Mat to fit SVM
		Mat resizedMat(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH, CV_8UC1, Scalar::all(0));
		resize(grayMat, resizedMat, resizedMat.size(), 0, 0, CV_INTER_AREA);
		//extract feature
		vector<int> feature = this->extractor.extract(resizedMat);
		return this->predicator.predict(feature);
	}

	SwitchState recognize(vector<Mat> const & imgPatches) const//recognize three patch
	{
		vector<SwitchState> states = this->recognizeEach(imgPatches);
		//decide final state
		map<SwitchState, int> counter;
		int max = INT_MIN;
		SwitchState modeState;
		for (int i = 0; i < STATES; i++)
		{
			counter[SwitchState(i)] = count(states.cbegin(), states.cend(), SwitchState(i));
			if (counter[SwitchState(i)]>max)
			{
				max = counter[SwitchState(i)];
				modeState = SwitchState(i);
			}
		}
		//decision logic
		if (counter[STATE_NEARLY_CLOSED] != 0)
			return STATE_NEARLY_CLOSED;
		//choose mode
		else
			return modeState;
	}

	vector<SwitchState> recognizeEach(vector<Mat> const & imgPatches) const
	{
		vector<SwitchState> states;
		for (Mat const& m : imgPatches)
			states.push_back(this->recognize(m));
		return states;
	}

};

/************************************************************************/
/* Utils: load indexFile                                         */
/************************************************************************/

vector<string> loadIndexFile(string indexFilePath)
{
	ifstream fs(indexFilePath);
	vector<string> result;
	if (fs.fail())
	{
		cout << "ERROR: can not open file!" << endl;
		exit(-1);
	}
	string line;//line-format:S005 001 3 3
	int lineNum = 0;
	while (getline(fs, line))
	{
		//if (lineNum >= 500)
		//{
		//	break;
		//}
		result.push_back(line);
		lineNum++;
	}
	return result;
}



int main()
{

	Predicator& pd = SvmPredicator(string("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\task\\cookeddata-0304\\cookeddata\\SVMs\\SVM.xml"),
		SIFT_FEATURE_DIMS);
	FeatureExtractor & fe = SiftFeatureExtractor(SIFT_KEY_POINTS);
	MultiStateRecognizer sr = MultiStateRecognizer(pd, fe);
	//image
	vector<string> indexFile = loadIndexFile("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\task\\cookeddata-0304\\cookeddata\\rp5\\tr.txt");
	string base = "C:\\Users\\guoqingpei\\Desktop\\newjingtai\\task\\cookeddata-0304\\cookeddata\\";
	//output precision to text-file
	ofstream outfile("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\task\\cookeddata-0304\\cookeddata\\Results\\testRecg.txt", ofstream::out);
	for (string const & img : indexFile)
	{
		stringstream s(img);
		string imgPath;
		int label;
		s >> imgPath >> label;
		Mat M = imread(base + imgPath, 0);
		SwitchState state = sr.recognize(M);
		cout << imgPath << "\t"
			<< label << "\t"
			<< state << endl;
	}

	//Mat M = imread("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\task\\cookeddata-0304\\cookeddata\\rp4/tr/30_33_0002672_0_01.jpeg");

	system("pause");
	return 0;
}
