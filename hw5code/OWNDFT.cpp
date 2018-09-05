#define _CRT_SECURE_NO_DEPRECATE //去除fopen報警告
#include <stdio.h> 
#include <cv.h> 
#include <highgui.h> 
#include <iostream>
#include <fstream>
#include <string>
using namespace cv;
using namespace std;

void OWNDFT(cv::Mat grayimage, int row, int cols)
{
	
	double degree;
	double cos ,sin;
	double RePixel, ImPixel;
	double P;
	int N=row; //y=v=N (範圍一樣)
	int M=cols; //x=u=M (範圍一樣)
	
	//f(x,y) -> 1D row transforms -> F(x,v)
	cv::Mat ReFxv(cols, row, CV_32F);
	cv::Mat ImFxv(cols, row, CV_32F);
	double valueReXV=0,valueImXV=0;		
	 //f(x,y) -> 1D row transforms -> F(x,v)
	for (int x = 0; x < M; x++)
    {
		for (int v = 0; v < N; v++)
        {
        //1D-DFT function        
            ((float*)ReFxv.data)[x*N+v]=0;
            ((float*)ImFxv.data)[x*N+v]=0;
										//	((float*)img3.data)[i*1000+j]
            for (int y = 0; y < N; y++)
            {
				 RePixel=((float)((uchar*)grayimage.data)[x*N+y])*pow(-1,(double)(x+y));// f(x,y):原圖    *(-1)^(x+y):做平移，讓亮點在中心
				 ImPixel=((float)((uchar*)grayimage.data)[x*N+y])*pow(-1,(double)(x+y));// f(x,y):原圖    *(-1)^(x+y):做平移，讓亮點在中心
				 degree = (double)(2.0 * 3.14 * v * y) / N; // 2 * PI * v * y / N
				 cos= cv::cos(degree);
				 sin= (cv::sin(degree))*(-1);

				((float*)ReFxv.data)[x*N+v] += (RePixel * cos);
                ((float*)ImFxv.data)[x*N+v] += (ImPixel * sin); 				
			}
			((float*)ReFxv.data)[x*N+v] = ((float*)ReFxv.data)[x*N+v] / (float)N;
            ((float*)ImFxv.data)[x*N+v] = ((float*)ImFxv.data)[x*N+v] / (float)N;
		}
	}

	
	cv::Mat ReFuv(cols, row, CV_32F);
	cv::Mat ImFuv(cols, row, CV_32F);

	//f(x,v) -> 1D column transforms -> F(u,v)	
	for (int u = 0; u < M; u++)
    {
		for (int v = 0; v < N; v++)
		{ 
		       
			((float*)ReFuv.data)[u*N+v]=0;
            ((float*)ImFuv.data)[u*N+v]=0;

            for (int x = 0; x < M; x++)
            {
                RePixel = ((float*)ReFxv.data)[x*N+v]; // F(x,v)
                ImPixel = ((float*)ImFxv.data)[x*N+v];
                degree = (double)(2.0 * 3.14 * u * x )/ M; // 2 * PI * u * x / M
				cos= cv::cos(degree);
				sin= (cv::sin(degree))*(-1);

                ((float*)ReFuv.data)[u*N+v] += (RePixel * cos - ImPixel * sin);
                ((float*)ImFuv.data)[u*N+v] += (ImPixel * cos + RePixel * sin);
            }
            ((float*)ReFuv.data)[u*N+v] = ((float*)ReFuv.data)[u*N+v] / (float)M;
            ((float*)ImFuv.data)[u*N+v] = ((float*)ImFuv.data)[u*N+v] / (float)M;
			
        }
    }
	double tanP , limitP , Maximum=0;
	cv::Mat DFTprocess_Spectrum_Magnitude(cols, row, CV_32F);
	cv::Mat DFTprocess_Spectrum_Phase(cols, row, CV_32F);
	for (int u = 0; u < M; u++)
    {
        for (int v = 0; v < N; v++)
        {
			//Magnitude
			P = sqrt(  ((float*)ReFuv.data)[u*N+v] * ((float*)ReFuv.data)[u*N+v] + ((float*)ImFuv.data)[u*N+v] * ((float*)ImFuv.data)[u*N+v]  );
			P = 1.0 * pow(P, 0.1) ;
			if(P<0)
				P=0.0;
			if(P>255)
				P=255.0;
			
			((float*)DFTprocess_Spectrum_Magnitude.data)[u*N+v]=P;

			//Phase
			tanP=(float)(atan( ((float*)ImFuv.data)[u*N+v]/((float*)ReFuv.data)[u*N+v] )*180.0/3.14);
			if(   ImFuv.ptr<float>(u)[v]>0   &&  ReFuv.ptr<float>(u)[v]==0   )	  //90度			
				tanP=90.0;
			if(   ImFuv.ptr<float>(u)[v]<0   &&  ReFuv.ptr<float>(u)[v]==0   )	  //270度
				tanP=270.0;
			if(   ImFuv.ptr<float>(u)[v]>0   &&  ReFuv.ptr<float>(u)[v]<0   )	//第二象限
				tanP=abs(tanP)+90.0;
			if(   ImFuv.ptr<float>(u)[v]<0   &&  ReFuv.ptr<float>(u)[v]<0   )	//第三象限
				tanP=tanP+180.0;
			if(   ImFuv.ptr<float>(u)[v]<0   &&  ReFuv.ptr<float>(u)[v]>0   )	//第四象限
				tanP=tanP+360.0;
																				//第一象限不用處理，角度求出來為正值(y軸、x軸都為正)
			//將角度範圍(0~360)轉為像數範圍(0~255)
			limitP = tanP / 360.0 * 255.0;
			((float*)DFTprocess_Spectrum_Phase.data)[u*N+v]=limitP;
		}
	}
	//將大小頻譜(float)先存起來，再轉換成 uchar 型別，再用Histogram Equalization去強化頻譜
	cv::Mat DFTprocess_Spectrum_Magnitude_uchar(cols, row, CV_8UC1);
	cv::Mat DFTprocess_Spectrum_Magnitude_HE(cols, row, CV_8UC1);
	for (int u = 0; u < M; u++)
    {
        for (int v = 0; v < N; v++)
        {
			DFTprocess_Spectrum_Magnitude_uchar.ptr<uchar>(u)[v]=(uchar)(DFTprocess_Spectrum_Magnitude.ptr<float>(u)[v]+0.5);
		}
	}
	cv::equalizeHist(DFTprocess_Spectrum_Magnitude_uchar,DFTprocess_Spectrum_Magnitude_HE);//Histogram Equalization(openCV)，只能讀uchar型別
	//將相位頻譜(float)先存起來，再轉換成 uchar 型別，再用Histogram Equalization去強化頻譜
	cv::Mat DFTprocess_Spectrum_Phase_uchar(cols, row, CV_8UC1);
	cv::Mat DFTprocess_Spectrum_Phase_HE(cols, row, CV_8UC1);
	for (int u = 0; u < M; u++)
    {
        for (int v = 0; v < N; v++)
        {
			DFTprocess_Spectrum_Phase_uchar.ptr<uchar>(u)[v]=(uchar)(DFTprocess_Spectrum_Phase.ptr<float>(u)[v]+0.5);
		}
	}
	cv::equalizeHist(DFTprocess_Spectrum_Phase_uchar,DFTprocess_Spectrum_Phase_HE);//Histogram Equalization(openCV)，只能讀uchar型別
      
       
	imshow("magnitude",DFTprocess_Spectrum_Magnitude);
	imshow("phase",DFTprocess_Spectrum_Phase_uchar);
	imwrite("magnitude.png",DFTprocess_Spectrum_Magnitude_HE);
	imwrite("phase.png",DFTprocess_Spectrum_Phase_HE);








}