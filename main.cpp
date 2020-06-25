/**
 * Copyright (c) 2016, Dengfeng Chai
 * Contact: chaidf@zju.edu.cn
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <bitset>
#include "RSS.h"

using namespace std;
using namespace cv;

/** \brief Command line tool for running RSS.
 * Usage:
 *   $ a.out --help
 *   Allowed options:
 *     -h [ --help ]                   produce help message
 *     -i [ --input ] arg              the file/folder to process 
 *     -o [ --output ] arg             the file/folder to process
 *     -d [ --dimension ] arg          2 for image / 3 for video
 *     -f [ --function ] arg           0 diff / 1 range
 *     -k [ --superpixels ] arg (=400) number of superpixles
 *     -la [ --lambda ] arg (=3)       lambda
 *     -laz [ --lambdaz ] arg (=3)     lambdaz
 *     -p [ --perturb-seeds ] arg (=0) perturb seeds: > 0 yes, = 0 no
 *     -l [ --parallel ] arg (=0)      parallel
 *     -g [ --sigma ] arg (=1)         sigma used for smoothing (no smoothing if zero)
 * \endcode
 * \author Dengfeng Chai
 */

void Label2Color(int *label, Mat img_fill)
{
    int num=16*16*16;
	vector<Vec3b> rgb(16 * 16 * 16);
	int sr = 0, sg = 0, sb = 0;
	for (int r = 0; r < 16; r++, sr += 9)
	for (int g = 0; g < 16; g++, sg += 9)
	for (int b = 0; b < 16; b++, sb += 9)
	{
		sr = sr % 16;
		sg = sg % 16;
		sb = sb % 16;
		uchar vr = sr * 16;
		uchar vg = sg * 16;
		uchar vb = sb * 16;
		rgb[((r * 16 + g) * 16) + b] = Vec3b(vr, vg, vb);
	}
	int idx = 0;
    std::cout << img_fill.rows << " " << img_fill.cols << "\n";
	for (int i = 0; i<img_fill.rows; i++)
	for (int j = 0; j<img_fill.cols; j++)
	{
		int fij = label[idx++];
        fij = fij % num;
		img_fill.at<Vec3b>(i, j) = rgb[fij];
	}
}

void Label2Color(int *label, vector<Mat> img_fill)
{
	vector<Vec3b> rgb(16 * 16 * 16);
	int sr = 0, sg = 0, sb = 0;
	for (int r = 0; r < 16; r++, sr += 9)
	for (int g = 0; g < 16; g++, sg += 9)
	for (int b = 0; b < 16; b++, sb += 9)
	{
		sr = sr % 16;
		sg = sg % 16;
		sb = sb % 16;
		uchar vr = sr * 16;
		uchar vg = sg * 16;
		uchar vb = sb * 16;
		rgb[((r * 16 + g) * 16) + b] = Vec3b(vr, vg, vb);
	}
	int idx = 0;
	for (int t = 0; t < img_fill.size(); t++)
	for (int i = 0; i<img_fill[t].rows; i++)
	for (int j = 0; j<img_fill[t].cols; j++)
	{
		int fij = label[idx++];
		img_fill[t].at<Vec3b>(i, j) = rgb[fij];
	}
}

void Label2Boundary(int *label, Mat img, Mat img_boundary)
{
	int rows = img.rows;
	int cols = img.cols;
	img.copyTo(img_boundary);

	Mat istaken = Mat::zeros(img.size(), CV_8U);

	for (int i = 0; i<rows; i++)
	for (int j = 0; j<cols; j++)
	{
		int np(0);
		int fij = label[i*cols + j];
		//cout << fij << endl;
		bool flag = false;
		int l = max(0, j - 1);
		int r = min(cols - 1, j + 1);
		int u = max(0, i - 1);
		int b = min(rows - 1, i + 1);
		for (int ii = u; ii <= b; ii++)
		for (int jj = l; jj <= r; jj++)
		{
			int fn = label[ii*cols + jj];
			if (0 == istaken.at<uchar>(ii, jj))
			{
				if (fij != fn) np++;
			}
		}
		if (np>1)
		{
			istaken.at<uchar>(i, j) = 255;
			img_boundary.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
		}
	}
}

void LoadVideo(string in_path, vector<string> &fileNames, vector<Mat> &images)
{
	Directory dir;
	fileNames = dir.GetListFiles(in_path, ".png", false);
	int n = fileNames.size();
    sort(fileNames.begin(), fileNames.end());
	for (int i = 0; i < n; i++)
	{
		string fileFullName = in_path + fileNames[i];

		Mat img = imread(fileFullName);
		images.push_back(img);
	}
}

void SaveVideo(string out_path, vector<string> &fileNames, vector<Mat> &imgsgs)
{
    int status;
    status = mkdir(out_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	for (int i = 0; i < fileNames.size(); i++)
	{
		string fileName = fileNames[i];
		Mat imgsg = imgsgs[i];
		string fileoutput = out_path + fileName;
		imwrite(fileoutput, imgsg);
	}
}

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t-h,--help\t\t Show this help message\n"
              << "\t-i,--input\t\t Specify the input path"
              << "\t-o,--output\t\t Specify the output path"
              << "\t-d,--dimension\t\t 2 for image / 3 for video"
              << "\t-f,--function\t\t 0 diff / 1 range"
              << "\t-k,--superpixels\t\t number of superpixles"
              << "\t-la,--lambda\t\t lambda"
              << "\t-laz,--lambdaz\t\t lambdaz"
              << "\t-p,--perturb-seeds\t\t perturb seeds: > 0 yes, = 0 no"
              << "\t-l,--parallel\t\t parallel"
              << "\t-g,--sigma\t\t sigma used for smoothing, 0 no smoothing"
              << std::endl;
}

int main(int argc, const char** argv) 
{
     if (argc < 3) {
        show_usage(argv[0]);
        return 1;
    }
    
    int dimension = 2;
    int func = 0;
    int superpixels = 200;
    float lambda = 3.0;
    float lambdaz = 3.0;
    int perturb = 0;
    int parallel = 0;
    float sigma = 1.0;

    string inputpath;
    string outputpath;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) 
        {
            show_usage(argv[0]);
            return 0;
        }
        else if ((arg == "-d") || (arg == "--dimension")) 
        {
            if (i + 1 < argc) 
            {
                dimension = atoi(argv[i+1]);
            } else 
            {
                std::cerr << "--superpixel or supervoxel not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-f") || (arg == "--function")) 
        {
            if (i + 1 < argc) 
            {
                func = atoi(argv[i+1]);
            } else 
            {
                std::cerr << "--cost function not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-i") || (arg == "--input")) 
        {
            if (i + 1 < argc) 
            {
                inputpath = argv[i+1];
            } else 
            {
                std::cerr << "--input path not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-o") || (arg == "--output")) 
        {
            if (i + 1 < argc) 
            {
                outputpath = argv[i+1];
            } else 
            {
                std::cerr << "--output path not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-k") || (arg == "--numsuperpixels")) 
        {
            if (i + 1 < argc) 
            {
                superpixels = atoi (argv[i+1]);
            } else 
            {
                std::cerr << "--number of superpixels not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-la") || (arg == "--lambda")) 
        {
            if (i + 1 < argc) 
            {
                lambda = atof (argv[i+1]);
            } else 
            {
                std::cerr << "--lambda not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-laz") || (arg == "--lambda")) 
        {
            if (i + 1 < argc) 
            {
                lambdaz = atof (argv[i+1]);
            } else 
            {
                std::cerr << "--lambdaz not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-p") || (arg == "--perturb")) 
        {
            if (i + 1 < argc) 
            {
                perturb = atoi (argv[i+1]);
            } else 
            {
                std::cerr << "--perturb flag not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-l") || (arg == "--parallel")) 
        {
            if (i + 1 < argc) 
            {
                parallel = atoi (argv[i+1]);
            } else 
            {
                std::cerr << "--parallel flag not specified." << std::endl;
                return 1;
            }  
        }
        else if ((arg == "-g") || (arg == "--sigma")) 
        {
            if (i + 1 < argc) 
            {
                sigma = atof (argv[i+1]);
            } else 
            {
                std::cerr << "--sigma not specified." << std::endl;
                return 1;
            }  
        }
    }

    if (dimension==2)
    {
        Mat image = imread(inputpath);
        Mat labels;
        
        if (sigma > 0.01) {
            int size = std::ceil(sigma*4) + 1;
            GaussianBlur(image, image, cv::Size (size, size), sigma, sigma);
        }
        int nrows = image.rows;
        int ncols = image.cols;
        int nbands=image.channels();
        int nchannels = nbands+1;
        int total = nrows*ncols*nchannels;

        Mat plane[nbands], res[nchannels];

        split(image, plane);
        res[0] = Mat::zeros(image.size(), CV_8U);
        for(int i=0;i<nbands;i++)
            res[i+1]=plane[i];

        Mat dst(image.size(), CV_8UC(nchannels));
        merge(res, nchannels, dst);

        uchar *dataND = new uchar[total];

        for (int i=0;i<nrows;i++)
            memcpy(dataND+i*ncols*nchannels,dst.data+i*dst.step[0],ncols*nchannels);

        labels=Mat::zeros(image.size(), CV_32SC1);
        int *label = (int*)labels.data;

        RSS sp;
        sp.Superpixels(dataND, label, nrows, ncols, nchannels, superpixels, func, lambda, perturb, parallel);

        Mat imagesg = cv::Mat::zeros(image.size(),CV_8UC3);

        Label2Color((int*)labels.data, imagesg);
        imwrite(outputpath, imagesg);

    }
    else if (dimension==3)
    {
        vector<string> fileNames;
        vector<Mat> images;
        vector<Mat> labels;
        vector<Mat> imagesvs;

        LoadVideo(inputpath, fileNames, images);

        int n = fileNames.size();
        for (int i = 0; i < n; i++)
        {
            cv::Mat imagesv = cv::Mat::zeros(images[i].size(),CV_8UC3);
            imagesvs.push_back(imagesv);
        }

        int rows = images[0].rows;
        int cols = images[0].cols;
        int frames = images.size();
        int nbands = images[0].channels();
        int nchannels = nbands+1;
        int numpixels = frames*rows*cols;
        int total = numpixels*nchannels;

        int *label = new int[numpixels];
        uchar *dataND = new uchar[total];

        clock_t start, finish;
        start = clock();
        for (int t = 0; t < frames; t++)
        {
            Mat img = images[t];
            Mat plane[nbands], res[nchannels];
            if (sigma > 0.01) {
                int size = std::ceil(sigma*4) + 1;
                GaussianBlur(img, img, cv::Size (size, size), sigma, sigma);

            }

            split(img, plane);
            res[0] = Mat::zeros(img.size(), CV_8U);
            for(int i=0;i<nbands;i++)
                res[i+1]=plane[i];

            Mat dst(img.size(), CV_8UC(nchannels));
            merge(res, nchannels, dst);
            int ncb=cols*nchannels;
            int nrcb=rows*ncb;
            uchar *pdata=dataND+t*nrcb;

            for (int i=0;i<rows;i++)
                memcpy(pdata+i*ncb,dst.data+i*dst.step[0],ncb);
        }

        RSS sv;
        sv.Supervoxels(dataND, label, frames, rows, cols, nchannels, superpixels, func, lambda, lambdaz, perturb, parallel);

        finish = clock();
        float duration = finish - start;

        string timefile = outputpath + "/time.txt";
        ofstream out(timefile.c_str());
        out << duration/CLOCKS_PER_SEC  << "\n";

        Label2Color(label, imagesvs);
        SaveVideo(outputpath, fileNames, imagesvs);
    }

    return 0;
}
