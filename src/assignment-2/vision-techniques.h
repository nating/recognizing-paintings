//
//  vision-techniques.h
//  assignment-1
//
//  The purpose of the functions in this file is to be able to implement vision techniques with a single function call, abstracting them to a higher level.
//  The functions all return copys of images, rather than the opencv norm of passing in output parameters (for developers who prefer it that way)
//  https://github.com/nating/visionwork/blob/master/assignment-1/src/assignment-1/vision-techniques.h
//
//  Created by Geoffrey Natin on 19/11/2017.
//  Copyright © 2017 nating. All rights reserved.
//

#ifndef vision_techniques_h
#define vision_techniques_h

using namespace cv;
using namespace std;

//-------------------------------------------- VISION TECHNIQUES -------------------------------------------------

//This function returns the a copy of the image passed converted to grayscale
Mat grayscale(Mat img){
    Mat grayscaleMat;
    cvtColor(img, grayscaleMat, CV_RGB2GRAY);
    return grayscaleMat;
}

//This function returns a copy of the grayscale image passed, converted to a binary image using adaptive thresholding
Mat adaptiveBinary(Mat grayscaleImage,int block_size,int offset, int output_value){
    Mat binaryMat;
    adaptiveThreshold( grayscaleImage, binaryMat, output_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, offset);
    return binaryMat;
}

//This function returns a copy of the grayscale image passed, converted to a binary image OTSU thresholding
Mat binary(Mat grayscaleImage,int block_size,int offset, int output_value){
    Mat binaryMat;
    int threshold = 128; //This shouldn't matter as OTSU is used, but the parameter is required.
    cv::threshold( grayscaleImage, binaryMat, threshold, 255, THRESH_BINARY | THRESH_OTSU );
    return binaryMat;
}

//This function returns a copy of the binary image passed, closed using an nxn element
Mat close(Mat binaryImage,int n){
    Mat closedMat;
    Mat n_by_n_element(n, n, CV_8U, Scalar(1));
    morphologyEx(binaryImage, closedMat, MORPH_CLOSE, n_by_n_element);
    return closedMat;
}

//This function returns a copy of the binary image passed, opened using an nxn element
Mat open(Mat binaryImage,int n){
    Mat openedMat;
    Mat n_by_n_element(n, n, CV_8U, Scalar(1));
    morphologyEx(binaryImage, openedMat, MORPH_OPEN, n_by_n_element);
    return openedMat;
}

//This function takes an image and the accepted color difference for pixels to be in the same region, returning a flood filled copy of the image
Mat floodFill( Mat img, int color_difference){
    CV_Assert( !img.empty() );
    Mat flooded = img.clone();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) ); //The floodFill function requires that the rows and columns are this length
    for(int y=0;y<flooded.rows;y++){
        for(int x=0;x<flooded.cols;x++){
            if(mask.at<uchar>(y+1, x+1)==0){
                Point point(x,y);
                Point3_<uchar>* p = img.ptr<Point3_<uchar>>(y,x);
                Scalar pointColour(p->x,p->y,p->z); //The first point of each segment is used as the color for the flood fill
                floodFill( flooded, mask, Point(x,y), pointColour, 0, Scalar::all(color_difference), Scalar::all(color_difference),4);
            }
        }
    }
    return flooded;
}

//This function takes an image and mean-shift parameters and returns a version of the image that has had mean shift segmentation performed on it
Mat meanShiftSegmentation(Mat img, int spatial_radius, int color_radius, int maximum_pyramid_level){
    Mat res = img.clone();
    pyrMeanShiftFiltering(img,res,spatial_radius,color_radius,maximum_pyramid_level);
    return res;
}

#endif /* vision_techniques_h */
