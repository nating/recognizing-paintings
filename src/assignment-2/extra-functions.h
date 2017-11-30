//
//  extra-functions.h
//  assignment-1
//
//  This file contains support functions for the 'Locate text on Notices' program.
//  https://github.com/nating/visionwork/blob/master/assignment-1/src/assignment-1/extra-functions.h
//
//  Created by Geoffrey Natin on 20/11/2017.
//  Copyright © 2017 nating. All rights reserved.
//

#ifndef extra_functions_h
#define extra_functions_h

using namespace cv;
using namespace std;

//------------------------------------------- EXTRA FUNCTIONS FOR VISION ------------------------------------------------------

//This class represents the bounding rectangle of a segment from a mean-shift-segmented image. It has the bounding rectangle and the color of the segment.
class segmentRectangle{
public:
    Scalar color;
    Rect rect;
};

//This function returns true if the two rectangles intersect
bool intersect(Rect r1, Rect r2){
    return ((r1 & r2).area()>0);
}

//This function returns true if the second rectangle is enclosed within the first
bool isEnclosing(Rect r1, Rect r2){
    return ( r2.tl().x > r1.tl().x && r2.tl().y > r1.tl().y && r2.br().x < r1.br().x && r2.br().y < r2.br().y );
}

//This function takes contours and its hierarchy and returns an array of the indexes of the 'contours with at least n children' within the hierarchy
vector<vector<Point>> getContoursWithNChildren(vector<vector<Point>> contours, vector<Vec4i> hierarchy,int n){
    vector<vector<Point>> newContours;
    //Find all components with more than 2 children
    if ( !contours.empty() && !hierarchy.empty() ) {
        // loop through the contours
        for ( int i=0; i<contours.size(); i++ ) {
            //If the contour has children, count them
            if(hierarchy[i][2]>0){
                vector<vector<Point>> children;
                //Starting with j as the first child, while there is a next child, move j on and up the count
                int j=hierarchy[i][2];
                while(hierarchy[j][0]>0){
                    children.push_back(contours[j]);
                    j = hierarchy[j][0];
                }
                //If the contour has more than 2 children, add it to newContours
                if (children.size()>n) {
                    newContours.push_back(contours[i]);
                }
            }
        }
    }
    return newContours;
}

//This function displays the n images from a Mat array in a window (all images must be of the same color space)
void display_images(string window_name, vector<Mat> images){
    cv::Mat window = images[0];
    for(int i=1;i<images.size();i++){
        cv::hconcat(window,images[i], window); // horizontally concatenate images together
    }
    namedWindow(window_name,cv::WINDOW_AUTOSIZE);
    imshow(window_name,window);
}

//This function takes an image & its contours and fills each contour with the average pixel value within the contour
Mat fillContours(Mat img, vector<vector<Point>> contours){
    Mat contoursMat = img.clone();
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            
            //Create mask to find average pixel value in original image
            Mat labels = cv::Mat::zeros(img.size(), CV_8UC1);
            drawContours(labels, contours, i, Scalar(255),CV_FILLED);
            Scalar average_color = mean(img, labels);
            
            //Fill the contour with the average value of its pixels
            drawContours( contoursMat, contours, i, average_color,CV_FILLED);
        }
    }
    return contoursMat;
}


//This function takes an image & its contours and fills each contour with the average pixel value within the contour
Mat fillContoursRandom(Mat img, vector<vector<Point>> contours){
    Mat contoursMat = img.clone();
    RNG rng = theRNG();
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            Scalar newVal(rng(256),rng(256),rng(256));
            //Fill the contour with the average value of its pixels
            drawContours( contoursMat, contours, i, newVal,CV_FILLED);
        }
    }
    return contoursMat;
}


#endif /* extra_functions_h */
