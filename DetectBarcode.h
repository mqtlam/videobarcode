/*!
 * Barcode Detection Algorithm using DCT
 *
 * Usage: ./DetectBarcode inputfile outputfile
 *
 * \author Michael Lam <mqtlam@cs.washington.edu>
 */

#ifndef _DETECTBARCODE_H
#define _DETECTBARCODE_H

#define ERROR_EPSILON 0.00001
#define PIXEL_ON 255
#define PIXEL_OFF 0

//! maximum labels for connected components step
#define MAX_LABELS 250 

#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

/*!
 * Print program usage.
 */
void printUsage(char** argv);

/*!
 * Run the bar code detection algorithm.
 *
 * \param 	input	grayscale image matrix
 * \return 	cropped image containing bar code
 */
Mat detectBarcode(Mat input);

/*!
 * Returns the bounding Rectangle of the largest connected component
 * given a binary image matrix.
 *
 * \note	This could have been substituted with a library 
 * 		method for faster processing, but openCV has 
 * 		no built-in bwlabel.
 *
 * \param	input	binary image matrix
 * \return	bounding Rectangle of largest connected component
 */
Rect largestConnectedComponent(Mat input);

/*!
 * Returns the coordinates of the four prior 8-neighbors of a pixel.
 * Used in largestConnectedComponent().
 * 
 * \param	input		binary image matrix
 * \param	row		current pixel row
 * \param	col		current pixel column
 * \param	neighbors	updated with coordinates of neighbors
 */
void getPriorNeighbors(Mat input, int row, int col, vector<Point>& neighbors);

/*!
 * Returns the labels of the given list of neighbor pixels.
 * Used in largestConnectedComponent().
 *
 * \param	input		binary image matrix
 * \param	neighbors	coordinates of neighbors
 * \param	labels		updated with labels of neighbors
 */
void getLabels(Mat input, vector<Point>& neighbors, vector<int>& labels);

#endif
