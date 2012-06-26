=====================================
BARCODE DETECTION ALGORITHM USING DCT
=====================================

------------
REQUIREMENTS
------------

- CMake
- OpenCV
- C++ compiler (e.g. g++ on linux)

--------------------------
SETUP AND RUN INSTRUCTIONS
--------------------------

This algorithm is written in C++ with the OpenCV library. It was compiled on a linux machine (Ubuntu 11.04) and compiled using CMake to assist in cross-platform compilation. The algorithm could be run on a Windows machine.

1. Make sure CMake and OpenCV are installed. Good tutorial: http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html
2. Run 'cmake .' in the directory containing the algorithm (DetectBarcode.cpp) and all the other files.
3. Run 'make' in the same directory to build the binary.
4. Run the algorithm using './DetectBarcode inputfile outputfile'. inputfile works with JPEG and output file should be a PNG.

The algorithm takes an input JPG file containing a bar code and outputs the bar code region to a PNG file. One can of course modify the program to print out the intermediate steps of the algorithm or return something different at the end.

------------------
ALGORITHM OVERVIEW
------------------

The algorithm is more or less an implementation of the paper attached, entitled "Locating 1-D Bar Codes in DCT-Domain." The main difference is the end step. After performing the morphological operation, our algorithm finds the largest connected component and returns the bounding rectangle around it.

Our algorithm has been shown to work well with barcodes in the horizontal orientation. The algorithm works well on large image sizes; the attached test image is an example.

----------------------------------
SUGGESTIONS FOR FUTURE IMPROVEMENT
----------------------------------

- Tune other parameters in the algorithm.
- Add a machine learning step to the algorithm.
- Consider ways to make algorithm work with rotated bar codes.
- Make an Android application for accessible application.
 
----------
REFERENCES
----------

CMake: http://cmake.org
OpenCV: http://opencv.willowgarage.com/wiki
OpenCV Linux Installation: http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html
OpenCV Linux Commandline Development: http://docs.opencv.org/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html

