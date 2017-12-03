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
#include "json.hpp" //https://github.com/nlohmann/json
#include <fstream>
#include "vision-techniques.h"
#include "extra-functions.h"

using json = nlohmann::json;
using namespace cv;
using namespace std;

//-------------------------------------------------- FUNCTIONS ---------------------------------------------------------

//This class represents a painting from a gallery.
class Painting{
public:
    string name;
    vector<Point> points;
};

//This class represents a gallery and its images.
class Gallery{
public:
    string name;
    vector<Painting> paintings;
};



//This function takes a Gallery corresponding to the input image and creates a copy of the original image with its paintings' names and locations drawn onto it
void create_output_image(string path_to_output_image,string path_to_original_image,Gallery gallery){
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
Mat getWallMask(Mat& img,Scalar color_difference){
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
                floodFill( img, mask, Point(x,y), newVal, &rect, color_difference,color_difference,4);
                int segment_size = rect.width*rect.height;
                if(segment_size>largest_segment){
                    largest_segment = segment_size;
                    wallColor = newVal;
                }
            }
        }
    }
    
    Mat wall_mask = cl.clone();
    inRange(img, wallColor, wallColor, wall_mask);
    return wall_mask;
}

vector<Rect> getRectsAroundContours(vector<vector<Point>> contours, vector<Vec4i> hierarchy){
    
    vector<Rect> rects;
    
    if ( !contours.empty() && !hierarchy.empty() ) {
        for ( int i=0; i<contours.size(); i++ ) {
            Rect bounder = boundingRect(contours[i]);
            rects.push_back(bounder);
        }
    }
    return rects;
}

Mat dilate(Mat img,int se_size,int shape=MORPH_RECT){
    Mat res;
    dilate(img,res,getStructuringElement(shape, Size(se_size,se_size)));
    return res;
}

Mat invert(Mat img){
    Mat res;
    bitwise_not(img, res);
    return res;
}

Mat medianFilter(Mat img, int blur_size){
    Mat res;
    medianBlur(img, res, blur_size);
    return res;
}

vector<Rect> getPaintingLocations(Mat img, vector<Rect> sub_images){
    
    vector<Rect> painting_locations;
    
    for(int i=0;i<sub_images.size();i++){
        
        //Following something along the lines of this right now... http://artsy.github.io/blog/2014/09/24/using-pattern-recognition-to-automatically-crop-framed-art/
        
        cout << "1" << endl;
        //Create sub image to work with
        Mat sub_image = Mat(img,sub_images[i]);
        cout << "2" << endl;
        
        //Convert sub image to grayscale
        Mat gray = grayscale(sub_image);
        
        //Dilate the grayscale image to get rid of the less important edges
        int structuring_element_size = 3; //TODO: test this value for the best results
        Mat dil = dilate(gray,structuring_element_size);
        
        //Perform median filter to preserve edges and remove noise (https://en.wikipedia.org/wiki/Median_filter)
        int blur_size = 7; //TODO: Test this value for the best results
        Mat med = medianFilter(dil, blur_size);
        
        //The blog post does some 'shrinking and enlarging' here?
        
        //Perform canny edges to get the edges in the image
        //Apparently a good way of finding the threshold for canny edges: https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
        Mat rubbish;
        double otsu_thresh_val = cv::threshold(med, rubbish, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        double high_thresh_val  = otsu_thresh_val, lower_thresh_val = otsu_thresh_val * 0.5;
        int low_threshold = 100;
        int high_threshold = 300;
        Mat edges;
        //Canny(med, edges, low_threshold, high_threshold);
        Canny(med, edges, lower_thresh_val, high_thresh_val);
        
        //Dilate the edges
        Mat dil2 = dilate(edges,structuring_element_size);
        
        vector<Mat> color = {dil2};
        display_images("Color",color);
        waitKey(0);
        
        //Mat can = Scalar::all(0);
        //img.copyTo(can,edges);
        
    }
    
    return sub_images;
}

string getPaintingMatch(Mat img, Rect painting_location){
    
    return "";
}

void getHoles(Mat img,string path_to_output_folder,string image_name){
    
    //Perform mean shift segmentation on the image
    int spatial_radius = 7;
    int color_radius = 13;
    int maximum_pyramid_level = 1;
    Mat meanS = meanShiftSegmentation(img, spatial_radius, color_radius, maximum_pyramid_level);
    
    //Get a mask of just the wallc
    Mat wall_mask = getWallMask(meanS,Scalar::all(2));
    
    //Dilate the wall mask to remove noise
    int structuring_element_size = 10;
    Mat dilated_wall_mask = dilate(wall_mask,structuring_element_size);
    
    //Invert the wall mask
    Mat inverted_wall_mask = invert(dilated_wall_mask);
    
    //Perform connected components on the inverted wall mask to find the non-wall components
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(inverted_wall_mask,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    
    //Get the locations of the sub images to check for paintings
    vector<Rect> sub_images = getRectsAroundContours(contours,hierarchy);
    
    //Get location of painting in each sub image
    vector<Rect> painting_locations = getPaintingLocations(img,sub_images);
    
    //Check if any potential painting locations contain a painting from our database
    for(int i=0;i<painting_locations.size();i++){
        if(getPaintingMatch(img,painting_locations[i])!=""){
            //Also put the name of the painting on the image here
            rectangle(img, painting_locations[i].tl(), painting_locations[i].br(), Scalar(255,255,255),3);
        }
    }
    
    
    for(int i=0;i<sub_images.size();i++){
        rectangle(img, sub_images[i].tl(), sub_images[i].br(), Scalar(255,255,255),3);
    }
    
    vector<Mat> color = {img};
    display_images("Color",color);
    waitKey(0);
}

Mat createWallMask(Mat img){
    
    Mat e = img.clone();
    
    //Do mean shift segmentation and get the color of each segment
    //meanShi
    Mat meanS = img.clone();

    //Find the color of the biggest segment
    
    //Create a mask where all pixels within a certain euclidean distance from the wall color are black
    
    //You know have a binary image (the mask)
    
    //Change all black segments that are below a certain size to white (they are not the wall)
    
    //Now you have an image where the black is the things that are not the wall
    
    //Do connected components analysis and find the holes that are inside the wall. These are the potential paintings.
    
    //Create mask to find average pixel value in original image
    Mat labels = cv::Mat::zeros(img.size(), CV_8UC1); //This creates a fully black image of the same size as the original image
    //drawContours(labels, contours, i, Scalar(255),CV_FILLED); //This draws white on the pixels in the contour on the mask.
    //Scalar average_color = mean(img, labels); // This gets the average color of the pixels in the img where the mask is white.
    
    return e;
    
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
    Mat painting_images[number_of_painting_images];
    vector<Gallery> galleries;
    
    //Read in ground truths
    vector<Gallery> gallery_ground_truths = read_gallery_ground_truths(path_to_gallery_ground_truths_json);
    
    //Process each gallery image
    for(int i=0;i<number_of_gallery_images;i++){
        //Read in the image
        string gallery_image_name = "Gallery"+to_string(i+1)+".jpg";
        gallery_images[i] = imread(path_to_gallery_images+gallery_image_name);
        if(gallery_images[i].empty()){ cout << "Image "+to_string(i)+" empty. Ending program." << endl; return -1; }
        
        getHoles(gallery_images[i],path_to_gallery_means,gallery_image_name);
        //Locate paintings
        
        //Recognise paintings
        
        //Overlay image names and locations on gallery images ---Shouldn't be in final program
        
    }
    
    //Compare recognised paintings to ground truths (calculate different metrics)
    
    
    return 0;
}

