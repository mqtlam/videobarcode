/*!
 * Barcode Detection Algorithm using DCT
 *
 * Usage: ./DetectBarcode inputfile outputfile
 *
 * \author Michael Lam <mqtlam@cs.washington.edu>
 */

#include "DetectBarcode.h"
#include "DisjointSets.h"

int main(int argc, char** argv)
{
	Mat result;
	Mat image;

	// argument checking
	if (argc != 3)
	{
		printUsage(argv);
		return -1;
	}

	image = imread(argv[1], 0); // 0 = load image as grayscale

	if (!image.data)
	{
		printf("No image data \n");
		return -1;
	}

	// run algorithm
	result = detectBarcode(image);

	//  write results
	imwrite(argv[2], result); 
	return 0;
}

void printUsage(char** argv)
{
	printf("USAGE: %s inputfile outputfile \n", argv[0]);
}

Mat detectBarcode(Mat input)
{
	//! vertical block size
	const int blockSizeVertical = 32;

	//! horizontal block size
	const int blockSizeHorizontal = 32;

	//! factor to downsample original image
	const int downsampleFactor = 2;

	//! emphasis weight factor
	const int kE = 4;

	//! de-emphasis weight factor
	const int kD = 1;

	//! morphology structure size
	const int morphStructSize = 24;

	//! binary threshold determined from PR curve
	const float bwThreshold = 0.4523;

	// downsample step
	resize(input, input, Size(input.cols/downsampleFactor, input.rows/downsampleFactor));

	// resize image to be divisible by block
	int extraRows = (input.rows % blockSizeVertical != 0) ? blockSizeVertical - input.rows % blockSizeVertical : 0;
	int extraCols = (input.cols % blockSizeHorizontal != 0) ? blockSizeHorizontal - input.cols % blockSizeHorizontal : 0;
	copyMakeBorder(input, input, 0, extraRows, 0, extraCols, BORDER_REPLICATE);

	Mat original = input.clone();

	// perform DCT and calculate average DCT block
	Mat averageDCTBlock = Mat::zeros(blockSizeVertical, blockSizeHorizontal, CV_32F);
	Mat dctImg = Mat::zeros(input.rows, input.cols, CV_32F);

	int i, j;
	int counter = 0;
	for (i = 0; i < input.rows; i += blockSizeVertical)
	{
		for (j = 0; j < input.cols; j += blockSizeHorizontal)
		{
			// perform DCT
			Mat blockDCT = Mat_<float>(input(Rect(j, i, blockSizeHorizontal, blockSizeVertical)));
			dct(blockDCT, blockDCT); // compute 2D DCT
			
			blockDCT.at<float>(0, 0) = 0;
			blockDCT = abs(blockDCT);
			Mat tmp = dctImg(Rect(j, i, blockSizeHorizontal, blockSizeVertical));
			blockDCT.copyTo(tmp); // update region of dctImg with blockDCT

			// calculate average DCT block
			averageDCTBlock = averageDCTBlock + blockDCT;

			counter++;
		}
	}
	averageDCTBlock = (1.0/counter) * averageDCTBlock;

	// group DCT block and compute weight matrix
	int f;
	int numMaxes = max(blockSizeVertical, blockSizeHorizontal);
	Mat c_fmax = Mat::zeros(1, numMaxes, CV_32F);
	Mat W = -kD * Mat::ones(blockSizeVertical, blockSizeHorizontal, CV_8S);

	for (f = 0; f < numMaxes; f++)
	{
		for (j = 0; j < f; j++)
		{
			if (f < blockSizeVertical && j < blockSizeHorizontal) // boundary check
			{
				c_fmax.at<float>(0, f) = max(c_fmax.at<float>(0, f), averageDCTBlock.at<float>(f, j));
			}
		}
		for (i = 0; i < f-1; i++)
		{
			if (i < blockSizeVertical && f < blockSizeHorizontal) // boundary check
			{
				c_fmax.at<float>(0, f) = max(c_fmax.at<float>(0, f), averageDCTBlock.at<float>(i, f));
			}
		}
	}

	for (i = 0; i < W.rows; i++)
	{
		for (j = 0; j < W.cols; j++)
		{
			for (f = 0; f < numMaxes; f++)
			{
				if (abs(averageDCTBlock.at<float>(i, j) - c_fmax.at<float>(0, f)) < ERROR_EPSILON)
				{
					W.at<schar>(i, j) = kE;
					break;
				}
			}
		}
	}
	
	// perform element multiplication with W
	int ii, jj, r, c;
	int maxElement = 0;
	Mat locationImg = Mat::zeros(input.rows, input.cols, CV_32F);
	for (r = 0; r < input.rows; r += blockSizeVertical)
	{
		for (c = 0; c < input.cols; c += blockSizeHorizontal)
		{
			double runningSum = 0;
			for (i = r, ii = 0; i < r + blockSizeVertical; i++, ii++)
			{
				for (j = c, jj = 0; j < c + blockSizeHorizontal; j++, j++)
				{
					runningSum += dctImg.at<float>(i, j) * (float) W.at<schar>(ii, jj);
				}
			}

			double response = max(runningSum, 0.0);
			maxElement = max(maxElement, (int) response);

			for (i = r; i < r + blockSizeVertical; i++)
			{
				for (j = c; j < c + blockSizeHorizontal; j++)
				{
					locationImg.at<float>(i, j) = response;
				}
			}
		}
	}
	input = (1.0/maxElement) * locationImg;

	// morphological step
	Mat element = getStructuringElement(MORPH_RECT, Size(2*morphStructSize+1, 2*morphStructSize+1), 
						Point(morphStructSize, morphStructSize));
	dilate(input, input, element);

	// thresholding step
	threshold(input, input, bwThreshold, PIXEL_ON, 0); // 0 = default binary thresholding

	// get largest connected component region 
	Rect roi = largestConnectedComponent(input);	

	// crop region of interest and return it
	input = original(roi);

	return input;
}

Rect largestConnectedComponent(Mat input)
{
	//! pixels for extra horizontal padding in result
	const int extraHorizontalPadding = 50;

	int i, r, c;
	int label = 1;
	Mat output = Mat::zeros(input.rows, input.cols, CV_8U);
	DisjointSets ds(MAX_LABELS);

	int counts[MAX_LABELS];
	int left[MAX_LABELS];
	int right[MAX_LABELS];
	int top[MAX_LABELS];
	int bottom[MAX_LABELS];
	for (i = 0; i < MAX_LABELS; i++)
	{
		counts[i] = 0;
		right[i] = bottom[i] = -1;
		left[i] = input.cols + 1;
		top[i] = input.rows + 1;
	}

	map<int, int> componentsCounts;
	map<int, int> componentsLeft;
	map<int, int> componentsRight;
	map<int, int> componentsTop;
	map<int, int> componentsBottom;

	// find all "components" in first pass of connected components algorithm

	for (r = 0; r < input.rows; r++)
	{
		for (c = 0; c < input.cols; c++)
		{
			int value = (int) input.at<float>(r, c);

			if (value == PIXEL_ON)
			{
				vector<Point2i> neighbors;
				getPriorNeighbors(input, r, c, neighbors);

				int M = MAX_LABELS + 1;
				if (neighbors.empty())
				{
					M = label;
					label++;
					if (label == MAX_LABELS)
						fprintf(stderr, "Connected Components: overflow error\n");	
				}
				else
				{
					vector<int> labels;
					getLabels(output, neighbors, labels);
					for (vector<int>::iterator it = labels.begin(); it != labels.end(); it++)
					{
						int label  = (int) *it;
						if (label < M)
						{
							M = label;
						}
					}
				}
				output.at<uchar>(r, c) = M;

				counts[M]++;
				left[M] = min(left[M], c);
				right[M] = max(right[M], c);
				top[M] = min(top[M], r);
				bottom[M] = max(bottom[M], r);

				vector<int> labels;
				getLabels(input, neighbors, labels);
				for (vector<int>::iterator it = labels.begin(); it != labels.end(); it++)
				{
					int X = (int) *it;
					if (X == M)
					{
						continue;
					}

					ds.Union(M, X);
				}
			}
		}
	}

	// consolidate components and find boundaries

	for (i = 0; i < label; i++)
	{
		int parent = ds.FindSet(i);

		if (componentsCounts.find(parent) != componentsCounts.end())
		{
			int componentsCountsCurrent = componentsCounts.find(parent)->second;
			int componentsLeftCurrent = componentsLeft.find(parent)->second;
			int componentsRightCurrent = componentsRight.find(parent)->second;
			int componentsTopCurrent = componentsTop.find(parent)->second;
			int componentsBottomCurrent = componentsBottom.find(parent)->second;
			
			componentsCounts.erase(parent);
			componentsLeft.erase(parent);
			componentsRight.erase(parent);
			componentsTop.erase(parent);
			componentsBottom.erase(parent);

			componentsCounts.insert(pair<int, int>(parent, componentsCountsCurrent + counts[i]));
			componentsLeft.insert(pair<int, int>(parent, min(left[i], componentsLeftCurrent)));
			componentsRight.insert(pair<int, int>(parent, max(right[i], componentsRightCurrent)));
			componentsTop.insert(pair<int, int>(parent, min(top[i], componentsTopCurrent)));
			componentsBottom.insert(pair<int, int>(parent, max(bottom[i], componentsBottomCurrent)));
		}
		else
		{
			componentsCounts.insert(pair<int, int>(parent, counts[i]));
			componentsLeft.insert(pair<int, int>(parent, left[i]));
			componentsRight.insert(pair<int, int>(parent, right[i]));
			componentsTop.insert(pair<int, int>(parent, top[i]));
			componentsBottom.insert(pair<int, int>(parent, bottom[i]));
		}
	}

	// find largest component

	int maxKey = -1;
	int maxValue = -1;
	for (map<int, int>::iterator it = componentsCounts.begin(); it != componentsCounts.end(); it++)
	{
		int value = it->second;
		if (maxValue < value)
		{
			maxKey = it->first;
			maxValue = value;
		}
	}

	// return bounding rectangle of component with extra horizontal padding
	
	int x = max(0, componentsLeft.find(maxKey)->second - extraHorizontalPadding);
	int y = max(0, componentsTop.find(maxKey)->second);
	int w = min(componentsRight.find(maxKey)->second + extraHorizontalPadding, input.cols) - x;
	int h = componentsBottom.find(maxKey)->second - y;

	return Rect(x, y, w, h);
}

void getPriorNeighbors(Mat input, int row, int col, vector<Point>& neighbors)
{
	Point2i coords[4];
	coords[0] = Point2i(row - 1, col + 1);	// northeast
	coords[1] = Point2i(row - 1, col);	// north
	coords[2] = Point2i(row - 1, col - 1);	// northwest
	coords[3] = Point2i(row, col - 1);	// west

	for (int i = 0; i < 4; i++)
	{
		if (coords[i].x >= 0 && coords[i].x < input.rows
			&& coords[i].y >= 0 && coords[i].y <= input.cols)
		{
			if (input.at<float>(coords[i].x, coords[i].y) > PIXEL_OFF)
			{
				neighbors.push_back(coords[i]);
			}
		}
	}
}

void getLabels(Mat input, vector<Point>& neighbors, vector<int>& labels)
{
	for (vector<Point2i>::iterator it = neighbors.begin(); it != neighbors.end(); it++)
	{
		Point2i coord = (Point2i) *it;
		labels.push_back((int) input.at<uchar>(coord.x, coord.y));
	}
}
