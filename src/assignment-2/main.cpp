/*
 main.cpp
 assignment-2
 
 The purpose of this file is to take in images of galleries, and to locate and recognise paintings in the images.
 https://github.com/nating/visionwork/blob/master/assignment-2/src/assignment-2/main.cpp
 
 Created by Geoffrey Natin on 23/11/2017.
 Copyright © 2017 nating. All rights reserved.
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "json.hpp" //https://github.com/nlohmann/json
#include <fstream>
#include "vision-techniques.h"
#include "extra-functions.h"

using json = nlohmann::json;
using namespace cv;
using namespace std;

//-------------------------------------------------- Classes ---------------------------------------------------------

//----------------------------------Classes for Generating ground truths------------------------------

//This class represents a painting from a gallery.
class Painting{
public:
    string name;
    string title;
    Mat image;
    vector<Point> points;
};

//This class represents a gallery and its images.
class Gallery{
public:
    string name;
    Mat image;
    vector<Painting> paintings;
};

//----------------------------------Classes for Recognition of paintings in galleries------------------------------

class RecognisedPainting{
public:
    string name;
    string title;
    Mat image;
    vector<Point> frameLocation;
    vector<Point2f> painting_corners;
    vector<Point> painting_points;
};

//This class represents the bounding rectangle of a segment from a mean-shift-segmented image. It has the bounding rectangle and the color of the segment.
class RecognisedGallery{
public:
    string name;
    Mat image;
    vector<RecognisedPainting> recognised_paintings;
};


//-------------------------------------------------- FUNCTIONS ---------------------------------------------------------


//This function takes a Gallery corresponding to the input image and creates a copy of the original image with its paintings' names and locations drawn onto it
void create_ground_truth_gallery_image(string path_to_output_image,string path_to_original_image,Gallery gallery){
    Mat output_image = imread(path_to_original_image);
    for(int i=0;i<gallery.paintings.size();i++){
        
        //Find the appropriate position to place the painting name
        int top = 2147483647;
        int bottom = 0;
        int left = 2147483647;
        int right = 0;
        for(int j=0;j<gallery.paintings[i].points.size();j++){
            top = min(gallery.paintings[i].points[j].y,top);
            bottom = max(gallery.paintings[i].points[j].y,bottom);
            left = min(gallery.paintings[i].points[j].x,left);
            right = max(gallery.paintings[i].points[j].x,right);
        }
        string text = gallery.paintings[i].name;
        int baseline = 0;
        Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 2,&baseline);
        
        //Outline and title the painting in the image
        rectangle(output_image, Point(left+(right-left)/2-text_size.width/2,top-text_size.height*2.5), Point(left+(right-left)/2+text_size.width/2,top), Scalar(255,255,255),CV_FILLED);
        putText(output_image, text, Point(left+(right-left)/2-text_size.width/2,top-text_size.height) , FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,0),2);
        polylines(output_image, gallery.paintings[i].points, true, Scalar(0,0,255),2);
    }
    
    //Save the output image
    imwrite(path_to_output_image, output_image);
}

//This function reads ground truths for notice images from a json and returns a vector of ground truths of each image. (each image's ground truths are a vector of rectangles) (It's a very specific function!)
vector<Gallery> read_gallery_ground_truths(string path_to_ground_truths_json){
    vector<Gallery> galleries;
    std::ifstream i(path_to_ground_truths_json);
    json j;
    i >> j;
    for (json::iterator it = j["galleries"].begin(); it != j["galleries"].end(); ++it) {
        Gallery g;
        string g_name = it.value()["name"];
        g.name = g_name;
        vector<Painting> ps;
        for (json::iterator z = it.value()["paintings"].begin(); z != it.value()["paintings"].end(); ++z) {
            Painting p;
            string p_name = z.value()["name"];
            p.name = p_name;
            string p_title = z.value()["title"];
            p.title = p_title;
            vector<Point> points;
            for (json::iterator t = z.value()["points"].begin(); t != z.value()["points"].end(); ++t) {
                points.push_back(Point(t.value()[0],t.value()[1]));
            }
            p.points = points;
            ps.push_back(p);
        }
        g.paintings = ps;
        galleries.push_back(g);
    }
    return galleries;
}

//Returns a version of the image where the wall is white and the rest of the image is black.
Mat getMaskOfLargestSegment(Mat& img,Scalar color_difference){
    CV_Assert( !img.empty() );
    RNG rng = theRNG();
    Mat cl = img.clone();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    Scalar wallColor;
    int largest_segment = 0;
    for( int y = 0; y < img.rows; y++ ){
        for( int x = 0; x < img.cols; x++ ){
            if( mask.at<uchar>(y+1, x+1) == 0 ){
                Scalar newVal( rng(256), rng(256), rng(256) );
                Rect rect;
                floodFill( cl, mask, Point(x,y), newVal, &rect, color_difference,color_difference,4);
                int segment_size = rect.width*rect.height;
                if(segment_size>largest_segment){
                    largest_segment = segment_size;
                    wallColor = newVal;
                }
            }
        }
    }
    
    Mat wall_mask = img.clone();
    inRange(cl, wallColor, wallColor, wall_mask);
    return wall_mask;
}

//Returns the bounding rectangles of the contours
vector<Rect> getBoundingRectsAroundContours(Mat img,vector<vector<Point>> contours){
    vector<Rect> rects;
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            Rect bounder = boundingRect(contours[i]);
            rects.push_back(bounder);
        }
    }
    return rects;
}

//Returns true if the painting does not take up the entire image, is bigger than 'width' * 'height', and it's contour takes up a percentage of its bounding rectangle's area
bool couldBePainting(Mat img,Rect bounder,vector<Point> contour,int width,int height,double area_percentage){
    
    //Check that the rect is smaller than the entire image and bigger than a certain size
    if(bounder.width*bounder.height<img.rows*img.cols && bounder.width*bounder.height>width*height){
        
        //Extra to remove floors when programming
        if(contourArea(contour)>bounder.width*bounder.height*.6){
            return true;
        }
    }
    return false;
}

//Returns the indexes of the contours whos bouding rectangles are bigger than 'min_width' * 'min_height' and whos take up more than min_area_percentage of their bounding rectangle
vector<vector<Point>> getPossiblePaintingContours(Mat img,vector<vector<Point>> contours, int min_width=150, int min_height=150, double min_area_percentage=.6){
    
    vector<vector<Point>> painting_contours;
    
    if ( !contours.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            Rect bounder = boundingRect(contours[i]);
            if( couldBePainting(img, bounder, contours[i], min_width, min_height, min_area_percentage)){ painting_contours.push_back(contours[i]); }
        }
    }
    return painting_contours;
}

//Returns a dilated version of the image
Mat dilate(Mat img,int se_size,int shape=MORPH_RECT){
    Mat res;
    dilate(img,res,getStructuringElement(shape, Size(se_size,se_size)));
    return res;
}

//Returns an eroded version of the image
Mat erode(Mat img,int se_size,int shape=MORPH_RECT){
    Mat res;
    erode(img,res,getStructuringElement(shape, Size(se_size,se_size)));
    return res;
}

//Returns an inverted version of the image
Mat invert(Mat img){
    Mat res;
    bitwise_not(img, res);
    return res;
}

//Returns a version of the image with a median filter having been applied to it
Mat medianFilter(Mat img, int blur_size){
    Mat res;
    medianBlur(img, res, blur_size);
    return res;
}

//Returns true if lines are similar enough in angle and position
bool linesAreEqual(const Vec4i& _l1, const Vec4i& _l2){
    Vec4i l1(_l1), l2(_l2);
    
    float length1 = sqrtf((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]));
    float length2 = sqrtf((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]));
    
    float product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1]);
    
    if (fabs(product / (length1 * length2)) < cos(CV_PI / 30))
        return false;
    
    float mx1 = (l1[0] + l1[2]) * 0.5f;
    float mx2 = (l2[0] + l2[2]) * 0.5f;
    
    float my1 = (l1[1] + l1[3]) * 0.5f;
    float my2 = (l2[1] + l2[3]) * 0.5f;
    float dist = sqrtf((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2));
    
    if (dist > std::max(length1, length2) * 0.5f)
        return false;
    
    return true;
}

//Returns the corners found in the image using Harris Corner detection with different parameters
vector<Point2f> harrisCornerDetection(Mat img, int max_corners, double quality, int minimum_distance){
    vector<Point2f> corners;
    goodFeaturesToTrack(img, corners, max_corners, quality, minimum_distance);
    return corners;
}

//Returns an edge version of the image (Found using Canny)
Mat cannyEdgeDetection(Mat img){
    Mat not_needed;
    //Apparently a good way of finding the threshold for canny edges: https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    double otsu_thresh_val = cv::threshold(img, not_needed, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    double high_thresh_val  = otsu_thresh_val, lower_thresh_val = otsu_thresh_val * 0.5;
    Mat res;
    Canny(img, res, lower_thresh_val, high_thresh_val);
    return res;
}

//Returns the lines found from performing HoughLinesP on the image
vector<Vec4i> geoffHoughLines(Mat img, int rho, double theta, int hough_threshold, int min_line_length, int min_line_gap){
    vector<Vec4i> lines;
    HoughLinesP(img, lines, rho, theta, hough_threshold, min_line_length, min_line_gap);
    return lines;
}

//Groups similar lines in angle and position together and returns this new vector of lines
vector<Vec4i> groupHoughLines(vector<Vec4i> lines){
    
    //Find number of groups of lines that are similar
    vector<int> labels;
    int numberOfLines = cv::partition(lines, labels, linesAreEqual);
    
    vector<Vec4i> groupedLines;
    //Group together all lines from the same group
    for(int j=0;j<numberOfLines;j++){
        int tlx = 2147483647; int tly = 2147483647; int brx = -1; int bry = -1;
        for(int k=0;k<labels.size();k++){
            if(labels[k]==j){
                tlx = min(tlx, lines[k][0]);
                tly = min(tly, lines[k][1]);
                brx = max(brx, lines[k][2]);
                bry = max(bry, lines[k][3]);
            }
        }
        groupedLines.push_back({tlx,tly,brx,bry});
    }
    
    return groupedLines;
}

//Returns the angle between two lines https://stackoverflow.com/questions/15888180/calculating-the-angle-between-points
float angleBetween(const Point &v1, const Point &v2) {
    
    //Get the length of both lines
    float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);
    
    //Get the dot product of the two lines
    float dot = v1.x * v2.x + v1.y * v2.y;
    
    float a = dot / (len1 * len2);
    
    if (a >= 1.0)
        return 0.0;
    else if (a <= -1.0)
        return CV_PI;
    else
        return acos(a); // 0..PI
}

//This function takes four corners and returns the corners in the order tl, tr, br, bl.
vector<Point2f> orderCorners(vector<Point2f> corners){
    
    struct sortY { bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.y < pt2.y);} } mySortY;
    struct sortX { bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.x < pt2.x);} } mySortX;
    std::sort(corners.begin(),corners.end(),mySortY);
    std::sort(corners.begin(),corners.begin()+2,mySortX);
    std::sort(corners.begin()+2,corners.end(),mySortX);
    
    return corners;
}

//Returns an Affine Transformed version of the image, where the 'corners' in the original image have been translated to the corners of the 'ideal_mat'.
Mat affineTransform(Mat img, vector<Point2f> corners, Mat ideal_mat){
    
    //Order input corners of the image tl, tr, br, bl
    corners = orderCorners(corners);
    
    //Create the FROM points
    Point2f inputQuad[4]; inputQuad[0] = corners[0]; inputQuad[1] = corners[1]; inputQuad[2] = corners[3]; inputQuad[3] = corners[2];
    
    //Create the TO points
    Point2f outputQuad[4]; outputQuad[0] = Point2f( 0,0 ); outputQuad[1] = Point2f(ideal_mat.cols-1,0); outputQuad[2] = Point2f(ideal_mat.cols-1,ideal_mat.rows-1); outputQuad[3] = Point2f( 0,ideal_mat.rows-1);
    
    //Get the Perspective Transform Matrix i.e. lambda
    Mat lambda( 2, 4, CV_32FC1 );
    lambda = Mat::zeros(ideal_mat.rows,ideal_mat.cols, ideal_mat.type() );
    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    
    // Apply the Perspective Transform to the image
    Mat res;
    warpPerspective(img,res,lambda,ideal_mat.size() );
    return res;
}

//Returns whether a match has a distance of more than 150
bool too_far(const DMatch &m){
    return m.distance > 150;
}

//Returns the value of the Histogram Comparison of the two images (using the Intersection method https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#comparehist)
double getHistogramMatch(Mat img0, Mat img1){
    
    //Get HSV versions of the images
    Mat img0_hsv, img1_hsv;
    cvtColor( img0, img0_hsv, COLOR_BGR2HSV );
    cvtColor( img0, img1_hsv, COLOR_BGR2HSV );
    
    //Use 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    
    //Set ranges for hue from 0 to 179 and saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    
    //Use the 0-th and 1-st channels of the histograms for the comparison
    int channels[] = { 0, 1 };
    
    // Create the Histograms for both images
    MatND img0_hist;
    calcHist( &img0_hsv, 1, channels, Mat(), img0_hist, 2, histSize, ranges, true, false );
    normalize( img0_hist, img0_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    
    MatND img1_hist;
    calcHist( &img1_hsv, 1, channels, Mat(), img1_hist, 2, histSize, ranges, true, false );
    normalize( img1_hist, img1_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    
    //Compare the two histograms
    return compareHist( img0_hist, img1_hist, 2);
}

//Returns the index of painting in 'paintings' that has the closest histogram to the affine transformed version of the sub-image of 'img' that is can be found at 'corners'
int getBestMatchingHistogramIndex(Mat img,vector<Point2f> corners, vector<Mat> paintings){
    
    double best_match = 0;
    double best_match_idx = -1;
    
    //For each painting, calculate how much of a match its histogram has to the sub-image
    for(int i=0;i<paintings.size();i++){
        
        //Affine transform the sub-image
        Mat trans = affineTransform(img, corners, paintings[i]);
        
        //Compare the histograms of the transformed sub-image and the painting
        double match = getHistogramMatch(trans,paintings[i]);
        
        //Update the best match
        if(match>best_match){ best_match = match; best_match_idx = i; }
    }
    
    return best_match_idx;
}

//Returns the index of painting in 'paintings' that has the closest histogram to the affine transformed version of the sub-image of 'img' that is can be found at 'corners'
int getBestMatchingHistogramIndex(Mat img,vector<Point2f> corners, vector<Painting> paintings){
    
    double best_match = 0;
    double best_match_idx = -1;
    
    //For each painting, calculate how much of a match its histogram has to the sub-image
    for(int i=0;i<paintings.size();i++){
        
        //Affine transform the sub-image
        Mat trans = affineTransform(img, corners, paintings[i].image);
        
        //Compare the histograms of the transformed sub-image and the painting
        double match = getHistogramMatch(trans,paintings[i].image);
        
        //Update the best match
        if(match>best_match){ best_match = match; best_match_idx = i; }
    }
    
    return best_match_idx;
}

//Returns a version of the image where the lines have been drawn on it in 'color' (extended to reach across the entire image)
Mat extendLinesAcrossImage(Mat img, vector<Vec4i> lines, Scalar color){
    
    Mat res = img.clone();
    
    for( size_t j = 0; j < lines.size(); j++ ){
        float angle = atan2(lines[j][1] - lines[j][3], lines[j][0] - lines[j][2]);
        
        angle = angle * 180 / CV_PI;
        
        int length = max(img.rows,img.cols);
        Point P1(lines[j][0], lines[j][1]);
        Point P2,P3;
        
        P2.x =  (int)round(P1.x + length * cos(angle * CV_PI / 180.0));
        P2.y =  (int)round(P1.y + length * sin(angle * CV_PI / 180.0));
        
        P3.x =  (int)round(P1.x - length * cos(angle * CV_PI / 180.0));
        P3.y =  (int)round(P1.y - length * sin(angle * CV_PI / 180.0));
        
        line(res, P3,P2, color, 10, 8 );
    }
    
    return res;
    
}


//Returns the points of the painting in 'img' in the frame represented by the one contour from 'mask'
vector<Point> getPaintingPoints(Mat img, Mat mask){
    
    vector<Point> points;
    
    //Erode the image to get rid of things attached to the frame
    int erosion_structuring_element_size = 60;
    Mat err = erode(mask,erosion_structuring_element_size);
    
    //Blur the image to smooth the lines of the frame
    int blur_size = 31;
    Mat med = medianFilter(err, blur_size);
    
    //Get an edge version of the image
    Mat edges = cannyEdgeDetection(med);
    
    //Find lines in the edge image
    vector<Vec4i> lines;
    int painting_ratio = max(img.rows,img.cols)*.1;
    HoughLinesP( edges, lines, 1, CV_PI/180, 0, painting_ratio*1.5, painting_ratio);
    
    //Draw extended version of the lines found to create a mask whos largest contour should be the frame (and should have very distinct corners)
    Mat sudoku = cv::Mat::zeros(img.size(), CV_8UC1); //The mask drawn will look like a nine squares of a sudoku puzzle. (The two vertical and two horizontal lines around the frame extended across the image in the mask)
    sudoku = extendLinesAcrossImage(sudoku,lines,Scalar(255));
    
    //Do connected components on the sudoku version of the image
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(invert(sudoku),contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    
    //If the mask does not look like we expect (like a sudoku puzzle) then give up at this point :(
    if(contours.size()<9){ return points; }
    
    //Find the largest contour from the sudoku version of the image
    int max_contour = 0;
    for(int j=0;j<contours.size();j++){
        if(contours[j].size() > contours[max_contour].size()){
            max_contour = j;
        }
    }
    
    return contours[max_contour];
}

//Returns the corners of the painting in 'img' in the frame represented by the one contour from 'mask'
vector<Point2f> getPaintingCorners(Mat img, Mat mask){
    
    vector<Point2f> corners;
    
    //Erode the image to get rid of things attached to the frame
    int erosion_structuring_element_size = 60;
    Mat err = erode(mask,erosion_structuring_element_size);
    
    //Blur the image to smooth the lines of the frame
    int blur_size = 31;
    Mat med = medianFilter(err, blur_size);
    
    //Get an edge version of the image
    Mat edges = cannyEdgeDetection(med);
    
    //Find lines in the edge image
    vector<Vec4i> lines;
    int painting_ratio = max(img.rows,img.cols)*.1;
    HoughLinesP( edges, lines, 1, CV_PI/180, 0, painting_ratio*1.5, painting_ratio);
    
    //Draw extended version of the lines found to create a mask whos largest contour should be the frame (and should have very distinct corners)
    Mat sudoku = cv::Mat::zeros(img.size(), CV_8UC1); //The mask drawn will look like a nine squares of a sudoku puzzle. (The two vertical and two horizontal lines around the frame extended across the image in the mask)
    sudoku = extendLinesAcrossImage(sudoku,lines,Scalar(255));
    
    //Do connected components on the sudoku version of the image
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(invert(sudoku),contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    
    //If the mask does not look like we expect (like a sudoku puzzle) then give up at this point :(
    if(contours.size()<9){ return corners; }
    
    //Find the largest contour from the sudoku version of the image
    int max_contour = 0;
    for(int j=0;j<contours.size();j++){
        if(contours[j].size() > contours[max_contour].size()){
            max_contour = j;
        }
    }
    
    //Draw a mask with just the painting contour in it
    Mat painting_contour = cv::Mat::zeros(img.size(), CV_8UC1);
    drawContours(painting_contour, contours, max_contour, Scalar(255),CV_FILLED);
    
    //Perform Corner detection on the painting contour
    double corner_quality = .001;
    int minimum_distance = 20;
    corners = harrisCornerDetection(painting_contour, 4, corner_quality, minimum_distance);
    
    
    //Create mask image of painting locations for ground truths
    Mat gts = cv::Mat::zeros(img.size(), CV_8UC1);
    corners = orderCorners(corners);
    vector<Point> ps;
    ps.push_back(corners[0]);
    ps.push_back(corners[1]);
    ps.push_back(corners[3]);
    ps.push_back(corners[2]);
    
        fillConvexPoly(gts, ps, Scalar(255));
    Mat aff_img;
    img.copyTo(aff_img,gts);
    
    vector<Mat> color = {aff_img};
    display_images("Gallery 0",color);
    waitKey(0);
    
    return corners;
}

//
vector<Point2f> translateCorners(vector<Point2f> corners, Point tl){
    
    vector<Point2f> translated;
    
    for(int i=0;i<corners.size();i++){
        Point2f newPos(corners[i].x+tl.x,corners[i].y+tl.y);
        translated.push_back(newPos);
    }
    
    return translated;
}

//
vector<Point> translatePoints(vector<Point> points, Point tl){
    
    vector<Point> translated;
    
    for(int i=0;i<points.size();i++){
        Point newPos(points[i].x+tl.x,points[i].y+tl.y);
        translated.push_back(newPos);
    }
    
    return translated;
}


//Returns the index of the painting int 'possible_paintings' that matches the painting found at the 'corners' in 'img' best
int matchPainting(Mat img,vector<Point2f> corners, vector<Painting> possible_paintings){
    
    //Check the features of the found painting against
    long highest_matches = 0;
    int best_match_idx = -1;
    
    //Check how many Scale Invarient Feature Transform matches the image has with each painting
    for(int j=0;j<possible_paintings.size();j++){
        
        Mat painting = possible_paintings[j].image;
        
        //Perform an Affine Transformation on the found painting
        Mat transformed_image = affineTransform(img, corners, painting);
        
        //Get features from the transformed image
        Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
        vector<KeyPoint> trans_keypoints;
        f2d->detect(transformed_image, trans_keypoints);
        Mat trans_descriptors;
        f2d->compute(transformed_image, trans_keypoints, trans_descriptors);
        
        //Get the features from the painting
        vector<KeyPoint> painting_keypoints;
        f2d->detect(painting, painting_keypoints);
        Mat painting_descriptors;
        f2d->compute(painting, painting_keypoints, painting_descriptors);
        
        //Find the matches between the features in the transformed image and the painting
        BFMatcher matcher = BFMatcher(NORM_L2, false);
        vector<DMatch> matches;
        matcher.match(trans_descriptors,painting_descriptors, matches);
        matches.erase(remove_if(matches.begin(),matches.end(),too_far),matches.end());
        
        //Update the highest_matches
        if(matches.size()>highest_matches){
            highest_matches = matches.size();
            best_match_idx = j;
        }
        
    }
    
    //If there is a best match, then return it
    if(best_match_idx>=0){ return best_match_idx; }
    
    //Otherwise, return the index of painting with the most similar histogram
    else{
        return getBestMatchingHistogramIndex(img,corners,possible_paintings);
    }
}


//Returns a RecognisedGallery that represents the gallery from the image
RecognisedGallery recogniseGallery(Mat img,vector<Painting> possible_paintings){
    
    RecognisedGallery gallery;
    gallery.image = img;
    
    //Perform mean shift segmentation on the image
    int spatial_radius = 7;
    int color_radius = 13;
    int maximum_pyramid_level = 1;
    Mat meanS = meanShiftSegmentation(img, spatial_radius, color_radius, maximum_pyramid_level);
    
    //Get a mask of just the wall in the gallery
    Mat wall_mask = getMaskOfLargestSegment(meanS,Scalar::all(2));
    
    //Dilate the wall mask to remove noise
    int structuring_element_size = 18;
    Mat dilated_wall_mask = dilate(wall_mask,structuring_element_size);
    
    //Invert the wall mask for finding possible painting components
    Mat inverted_wall_mask = invert(dilated_wall_mask);
    
    //Perform connected components on the inverted wall mask to find the non-wall components
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(inverted_wall_mask,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    
    //Find the contours that are rectangular, big enough to be considered and not spanning the entire image
    vector<vector<Point>> frame_contours = getPossiblePaintingContours(inverted_wall_mask, contours);
    
    //For each frame contour, recognise a painting from it.
    for(int i=0;i<frame_contours.size();i++){
        RecognisedPainting r;
        Rect sub_image_location = boundingRect(frame_contours[i]);
        vector<Point2f> painting_corners = getPaintingCorners(Mat(img,sub_image_location),Mat(inverted_wall_mask,sub_image_location));
        vector<Point> painting_points = getPaintingPoints(Mat(img,sub_image_location),Mat(inverted_wall_mask,sub_image_location));
        
        //If the painting's corners have been found, then match it with a painting from our image bank
        if(painting_corners.size()==4){
            int painting_idx = matchPainting(Mat(img,sub_image_location),painting_corners,possible_paintings);
            r.image = possible_paintings[painting_idx].image;
            r.name = possible_paintings[painting_idx].name;
            r.title = possible_paintings[painting_idx].title;
            r.frameLocation = frame_contours[i];
            r.painting_points = translatePoints(painting_points,sub_image_location.tl());
            r.painting_corners = translateCorners(painting_corners,sub_image_location.tl());
            gallery.recognised_paintings.push_back(r);
        }
        
    }
    
    return gallery;
}

//Finds the DICE Coefficient, of a gallery image. Given the rectangles corresponding to detected text paintings and ground truths
//Pre-requistite: No detected paintings must overlap and no ground truths must overlap
double getDiceCoefficient(Mat img, vector<RecognisedPainting> rps, vector<Painting> gts){
    
    /*
     "The DICE cooefficient is 2 times the Area of Overlap
     (between the ground truth and the regions found by your program)
     divided by the sum of the Area of the Ground Truth and the Area of the regions found by your program." - Ken
    */
    
    //Create mask image of painting locations for recognised paintings
    Mat recognised_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    Mat ground_truth_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    for(int i=0;i<rps.size();i++){
        vector<vector<Point>> contours; contours.push_back(rps[i].painting_points);
        drawContours(recognised_mask, contours, 0, Scalar(200),CV_FILLED);
    }
    
    //Create mask image of painting locations for ground truths
    for(int i=0;i<gts.size();i++){
        vector<Point> ps;
        for(int j=0;j<gts[i].points.size();j++){
            ps.push_back(gts[i].points[j]);
        }
        fillConvexPoly(ground_truth_mask, ps, Scalar(200));
    }
    
    //Count the non-zero pixels in the masks
    Mat and_mask;
    bitwise_and(ground_truth_mask, recognised_mask, and_mask);
    double overlap = cv::countNonZero(and_mask);
    double rec_area = countNonZero(recognised_mask);
    double gt_area = countNonZero(ground_truth_mask);
    
    return (2*overlap) / (rec_area + gt_area);
}

//------------------------------------------------ MAIN PROGRAM --------------------------------------------------------

int main(int argc, char** argv){
    
    //Define filepaths
    string path_to_gallery_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/recognizing-paintings/assets/galleries/";
    string path_to_gallery_means = "/Users/GeoffreyNatin/Documents/GithubRepositories/recognizing-paintings/assets/mean-shifts/sr_80_color_radius_65/";
    string path_to_painting_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/recognizing-paintings/assets/paintings/";
    string path_to_output_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/recognizing-paintings/assets/output_images/";
    string path_to_gallery_ground_truths_json = "/Users/GeoffreyNatin/Documents/GithubRepositories/recognizing-paintings/assets/painting-positions.json";
    
    //Initialise variables
    const int number_of_gallery_images = 4;
    const int number_of_painting_images = 6;
    Mat gallery_images[number_of_gallery_images];
    vector<Painting> paintings;
    vector<RecognisedGallery> recognised_galleries;
    
    //Read in ground truths
    vector<Gallery> gallery_ground_truths = read_gallery_ground_truths(path_to_gallery_ground_truths_json);
    
    //Create the Painting objects to be able to recognise
    for(int i=0;i<number_of_painting_images;i++){
        string painting_name = "Painting"+to_string(i+1)+".jpg";
        Mat painting_image = imread(path_to_painting_images+painting_name);
        Painting p;
        p.image = painting_image;
        p.name = painting_name;
        paintings.push_back(p);
    }
    
    //Get the title for each painting
    for(int k=0;k<paintings.size();k++){
        for(int i=0;i<gallery_ground_truths.size();i++){
            for(int j=0;j<gallery_ground_truths[i].paintings.size();j++){
                if(gallery_ground_truths[i].paintings[j].name==paintings[k].name){
                    paintings[k].title = gallery_ground_truths[i].paintings[j].title;
                }
            }
        }
    }
    
    //Process each gallery image
    for(int i=0;i<number_of_gallery_images;i++){
        //Read in the image
        string gallery_image_name = "Gallery"+to_string(i+1)+".jpg";
        gallery_images[i] = imread(path_to_gallery_images+gallery_image_name);
        if(gallery_images[i].empty()){ cout << "Image "+to_string(i)+" empty. Ending program." << endl; return -1; }
        
        RecognisedGallery g = recogniseGallery(gallery_images[i],paintings);
        
        //Draw an image of the paintings having been recognised in the gallery
        Mat rec = gallery_images[i].clone();
        for(int j=0;j<g.recognised_paintings.size();j++){
            
            int top = 2147483647;
            int bottom = 0;
            int left = 2147483647;
            int right = 0;
            for(int k=0;k<g.recognised_paintings[j].frameLocation.size();k++){
                top = min(g.recognised_paintings[j].frameLocation[k].y,top);
                bottom = max(g.recognised_paintings[j].frameLocation[k].y,bottom);
                left = min(g.recognised_paintings[j].frameLocation[k].x,left);
                right = max(g.recognised_paintings[j].frameLocation[k].x,right);
            }
            
            string text = g.recognised_paintings[j].name;
            int baseline = 0;
            Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 1, 2,&baseline);
            putText(rec, text, Point(left+(right-left)/2-text_size.width/2,top-text_size.height) , FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,0),2);
            
            vector<Point2f> cs = orderCorners(g.recognised_paintings[j].painting_corners);
            line(rec,cs[0],cs[1],Scalar(0,0,255),5);
            line(rec,cs[1],cs[3],Scalar(0,0,255),5);
            line(rec,cs[2],cs[3],Scalar(0,0,255),5);
            line(rec,cs[2],cs[0],Scalar(0,0,255),5);
            
            /*
            vector<vector<Point>> contours; contours.push_back(g.recognised_paintings[j].painting_points);
            drawContours(rec, contours, 0, Scalar(0,255,0),CV_FILLED);
            */
        }
        
        vector<Mat> color = {rec};
        display_images("Gallery 0",color);
        waitKey(0);
         
        
        g.name = "Gallery"+to_string(i);
        recognised_galleries.push_back(g);
    }
    
    //Compare recognised paintings to ground truths (calculate different metrics)
    
    //Count the true positives, false positives, and false negatives
    int true_positives = 0;
    int false_negatives = 0;
    int false_positives = 0;
    
    //For every gallery, count how many of its paintings were found correctly and how many were not found
    for(int i=0;i<gallery_ground_truths.size();i++){
        
        //Check if each of its images were found
        for(int k=0;k<gallery_ground_truths[i].paintings.size();k++){
            
            bool present = false;
            //Check that the painting is present in the recognised gallery
            for(int l=0;l<recognised_galleries[i].recognised_paintings.size();l++){
                if(gallery_ground_truths[i].paintings[k].name==recognised_galleries[i].recognised_paintings[l].name){ present = true; }
            }
            if(present){ true_positives++; }
            else{ false_negatives++; }
            
        }
    }
    
    //Count how many paintings were recognised that aren't present in the ground truth galleries
    for(int i=0;i<recognised_galleries.size();i++){
        
        //Count the found paintings that are not actually in the ground truth gallery
        for(int k=0;k<recognised_galleries[i].recognised_paintings.size();k++){
            
            bool present = false;
            //Check that the painting is present in the recognised gallery
            for(int l=0;l<gallery_ground_truths[i].paintings.size();l++){
                if(gallery_ground_truths[i].paintings[l].name==recognised_galleries[i].recognised_paintings[k].name){ present = true; break; }
            }
            if(!present){ false_positives++; }

        }
    }
    
    //Calculate Dice Coefficients for each gallery image
    double die[number_of_gallery_images];
    for(int i=0;i<gallery_ground_truths.size();i++){
        die[i] = getDiceCoefficient(gallery_images[i],recognised_galleries[i].recognised_paintings,gallery_ground_truths[i].paintings);
        printf("Gallery: %d, Dice Coefficient: %f.\n",i,die[i]);
    }
    
    printf("True Positives: %d, False Negatives: %d, False Positives: %d.",true_positives,false_negatives,false_positives);
    
    return 0;
}

