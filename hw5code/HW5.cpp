#define _CRT_SECURE_NO_DEPRECATE //去除fopen報警告
#include <stdio.h> 
#include <cv.h> 
#include <highgui.h> 
#include <iostream>
#include <fstream>
#include <string>
using namespace cv;
using namespace std;
void OWNDFT(cv::Mat grayimage, int row, int cols);
Mat opencv_dft(cv::Mat grayimage)  ;  
Mat IDFT(Mat src);




void main()
{

unsigned char *blackwhite ,*sine4 ,*sine16 ,*clown ,*mask;
char FileNameOri[]="blackwhite_256.raw";
char FileNamebright[]="sine4_128x128.raw";
char FileNamedark[]="sine16_128x128.raw";
char FileNameearth[]="clown_128x128.raw";
char FileNamemask[]="clown_mask_128x128.raw";
blackwhite = new unsigned char[256*256]; 
sine4 = new unsigned char[128*128]; 
sine16 = new unsigned char[128*128]; 
clown = new unsigned char[128*128];
mask = new unsigned char[128*128]; 
CvMat *BWmat = cvCreateMat(256,256, CV_8UC1);
CvMat *sine4mat = cvCreateMat(128,128, CV_8UC1);
CvMat *sine16mat = cvCreateMat(128,128, CV_8UC1);
CvMat *clownmat = cvCreateMat(128,128, CV_8UC1);
CvMat *maskmat = cvCreateMat(128,128, CV_8UC1);
FILE *BW256;
	BW256 = fopen(FileNameOri,"rb");
FILE *sine4_128;
	sine4_128 = fopen(FileNamebright,"rb");
FILE *sine16_128;
	sine16_128 = fopen(FileNamedark,"rb");
FILE *clown128;
	clown128 = fopen(FileNameearth,"rb");
FILE *mask128;
	mask128 = fopen(FileNamemask,"rb");

		fread(blackwhite,1,256*256,BW256);
		cvSetData(BWmat,blackwhite,BWmat->step);
		fread(sine4,1,128*128,sine4_128);
		cvSetData(sine4mat,sine4,sine4mat->step);
		fread(sine16,1,128*128,sine16_128);
		cvSetData(sine16mat,sine16,sine16mat->step);
		fread(clown,1,128*128,clown128);
		cvSetData(clownmat,clown,clownmat->step);
		fread(mask,1,128*128,mask128);
		cvSetData(maskmat,mask,maskmat->step);


cv::Mat dst;
dst = Mat(BWmat->rows,BWmat ->cols,BWmat->type,BWmat->data.fl);//轉換格式
cv::Mat sine4_128128;
sine4_128128 = Mat(sine4mat->rows,sine4mat ->cols,sine4mat->type,sine4mat->data.fl);//轉換格式
cv::Mat sine16_128128;
sine16_128128 = Mat(sine16mat->rows,sine16mat ->cols,sine16mat->type,sine16mat->data.fl);//轉換格式
cv::Mat clown_128128;
clown_128128 = Mat(clownmat->rows,clownmat ->cols,clownmat->type,clownmat->data.fl);//轉換格式		
cv::Mat clown_mask;
clown_mask = Mat(maskmat->rows,maskmat ->cols,maskmat->type,maskmat->data.fl);//轉換格式		



	int number1=0;
	cout<<"輸入題號:例如1_a小題為11 2_a小題為21";
	cin>> number1;
	if( number1 ==1)
	{ OWNDFT(dst,256,256);

		cvWaitKey(0);
	}	

	if(number1==21)
	{
	opencv_dft(sine4_128128);
	imshow("sine4",opencv_dft(sine4_128128));  
	Mat sine4;
    normalize(opencv_dft(sine4_128128), sine4, 0, 255, CV_MINMAX);                                                           // 规范化值到 0~1 显示图片的需要  
	imwrite("sine4.png",sine4);    

	opencv_dft(sine16_128128);
	imshow("sine16",opencv_dft(sine16_128128));
	Mat sine16;
	normalize(opencv_dft(sine16_128128), sine16, 0, 255, CV_MINMAX);   
	imwrite("sine16.png",sine16);    
		cvWaitKey(0);
	}

	if(number1==22)
	{
	opencv_dft(clown_128128);
	Mat test1; 
		Mat clown;
		normalize(opencv_dft(clown_128128), clown, 0, 1, CV_MINMAX); 
    normalize(opencv_dft(clown_128128), test1, 0, 255, CV_MINMAX);  
	imshow("clown_128",clown);                                                         
		imwrite("clown.png",test1); 


	//Mat newclown(128,128,CV_8U);
	//int test;
	//for(int i=0;i<127;i++)
	//{
	//	for(int j=0;j<127;j++)
	//	{    
	//	test=clown.at<float>(i,j)*255;
	//	newclown.at<uchar>(i,j)=test;
	//	}
	//}

	//imshow("newclown",newclown);
	Mat clownXmask(128,128,CV_8U);
	for(int i=0;i<128;i++)
	{
		for(int j=0;j<128;j++)
		{
			clownXmask.at<uchar>(i,j)=clown.at<float>(i,j)*clown_mask.at<uchar>(i,j);
			if (clownXmask.at<uchar>(i,j)>255)
				clownXmask.at<uchar>(i,j)=255;
			if(clownXmask.at<uchar>(i,j)<0)
				clownXmask.at<uchar>(i,j)=0;
		}
	}
	imwrite("clownXmask.png",clownXmask);
	imshow("clownXmask",clownXmask);


	
		cvWaitKey(0);
	}


	if(number1=23)
	{
		Mat I = clown_128128;
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

		Mat newmask;
	clown_mask.convertTo(newmask,CV_32F);



	Mat complex1;
	complex1=planes[0].clone();
	Mat complex2;
	complex2=planes[1].clone();
	for(int i=0;i<128;i++)
	{
		for(int j =0;j<128;j++)
		{
			planes[0].at<float>(i,j)=planes[0].at<float>(i,j)*(newmask.at<float>(i,j)/255);
			planes[1].at<float>(i,j)=planes[1].at<float>(i,j)*(newmask.at<float>(i,j)/255);
		}
	}
	//imshow("00",planes[0]);
	 merge(planes, 2, complexI);  
	 	imshow("10000000",planes[0]);
	imshow("11111111",planes[1]);

    //magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude

    //Mat magI = planes[0];

    //magI += Scalar::all(1);                    // switch to logarithmic scale
    //log(magI, magI);

    //// crop the spectrum, if it has an odd number of rows or columns
    //magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    //// rearrange the quadrants of Fourier image  so that the origin is at the image center
    //int cx = magI.cols/2;
    //int cy = magI.rows/2;

    //Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    //Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    //Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    //Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    //Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    //q0.copyTo(tmp);
    //q3.copyTo(q0);
    //tmp.copyTo(q3);

    //q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    //q2.copyTo(q1);
    //tmp.copyTo(q2);


    //normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	



    

    //calculating the idft
    cv::Mat inverseTransform;
	cv::Mat newinverseTransform;
    cv::idft(complexI, inverseTransform, cv::DFT_SCALE|cv::DFT_REAL_OUTPUT);
	inverseTransform.convertTo(newinverseTransform,CV_8U);
   // normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
	    imshow("original", clown_128128);
    imshow("Reconstructed", newinverseTransform);
	imwrite("IDFTTTT.png",newinverseTransform);
    waitKey(0);




	}







	}
