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
#define TRAIN_IMAGE_WIDTH 536  
#define TRAIN_IMAGE_HEIGHT 188
#define SIFT_KEY_POINTS 40
#define SIFT_DESCRIPTOR_DIMS 128
#define SIFT_FEATURE_DIMS SIFT_KEY_POINTS*SIFT_DESCRIPTOR_DIMS

/*Recognize Result*/
typedef enum SwitchState
{
	STATE_FULLY_OPEN,
	STATE_FULLY_CLOSED,
	STATE_HALF_OPEN,
	STATE_NEARLY_CLOSED,
}SwitchState;


/*feature extractor*/
class FeatureExtractor
{
public:
	virtual vector<float> extract(Mat const & image) const = 0;
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

	vector<float> extract(Mat const & image) const override
	{
		vector<KeyPoint> keypoints;
		Mat descriptors;
		vector<float> feature(expectKeyPoints*SIFT_DESCRIPTOR_DIMS,0.0);
		sift(image, Mat(), keypoints, descriptors, false);
		for (int i = 0; i < descriptors.rows; i++)
		{
			for (int j = 0; j < SIFT_DESCRIPTOR_DIMS; j++)
			{
				float val = descriptors.at<float>(i, j);
				feature[i*descriptors.rows+j] = val;
				//cout << val << "\t";
			}
		}
		return feature;
	}
};


/*Recognize State*/
class Predicator
{
public:
	virtual SwitchState predict(Mat const &feature) const = 0;
	virtual SwitchState predict(vector<float> const & feature) const = 0;
};

class SvmPredicator : public Predicator
{
private: 
	string svmPath;//svmPath: path to save trained SVMs
	int featureDim;//Size: feature size for SVM predict 
	CvSVM SVM;
public:	
	SvmPredicator(string svmPath, int featureDim) :svmPath(svmPath), featureDim(featureDim){
		SVM.load(svmPath.c_str());}//loadSVM

	SwitchState predict(Mat const & feature) const override
	{
		assert(feature.rows == 1 && feature.cols == featureDim);
		//SVM predict
		int predictLabel = SVM.predict(feature);
		return SwitchState(predictLabel);
	}

	SwitchState predict(vector<float> const & feature)const override
	{
	/*	vector<float> vec(feature.begin(),feature.end());*/
		Mat sampleMat = Mat(feature).reshape(1, 1);
		//Mat sampleMat(1, SIFT_FEATURE_DIMS, CV_32F, cv::Scalar::all(0.0));
		//for (int j = 0; j < SIFT_FEATURE_DIMS; j++)
		//{
		//	sampleMat.at<float>(j) = feature[j];
		//}
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
		//resize current Mat to fit SVM
		Mat resizedMat(TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH,CV_8UC1, Scalar::all(0));
		resize(grayMat, resizedMat, resizedMat.size(), 0, 0, CV_INTER_AREA);
		//extract feature
		vector<float> feature = this->extractor.extract(resizedMat);
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



int main()
{
	Predicator& pd = SvmPredicator(string("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\cookeddata\\SVMs\\Sift_SVM_1v8.xml"),
					SIFT_FEATURE_DIMS);
	FeatureExtractor & fe = SiftFeatureExtractor(SIFT_KEY_POINTS);
	MultiStateRecognizer sr = MultiStateRecognizer(pd, fe);
	//image
	Mat M = imread("C:\\Users\\guoqingpei\\Desktop\\newjingtai\\cookeddata\\rp\\tr\\30_33_0002792_0_00.jpeg");
	SwitchState state = sr.recognize(M);
	cout << state << endl;
	system("pause");
}
