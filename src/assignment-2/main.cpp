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
                cout << t.value() << endl;
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

//------------------------------------------------ MAIN PROGRAM --------------------------------------------------------

int main(int argc, char** argv){
    
    //Define filepaths
    string path_to_gallery_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-2/assets/galleries/";
    string path_to_painting_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-2/assets/paintings/";
    string path_to_output_images = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-2/assets/output_images/";
    string path_to_gallery_ground_truths_json = "/Users/GeoffreyNatin/Documents/GithubRepositories/visionwork/assignment-2/assets/painting-positions.json";
    
    //Initialise variables
    const int number_of_gallery_images = 4;
    const int number_of_painting_images = 6;
    Mat gallery_images[number_of_gallery_images];
    Mat painting_images[number_of_painting_images];
    vector<Gallery> galleries;
    
    //Read in ground truths
    vector<Gallery> gallery_ground_truths = read_gallery_ground_truths(path_to_gallery_ground_truths_json);
    
    //Create vector of paintings that the program can recognise
    for(int i=0;i<number_of_painting_images;i++){
        string painting_image_name = "Painting"+to_string(i+1)+".jpg";
        painting_images[i] = imread(path_to_painting_images+painting_image_name);
    }
    
    //Process each gallery image
    for(int i=0;i<number_of_gallery_images;i++){
        //Read in the image
        string gallery_image_name = "Gallery"+to_string(i+1)+".jpg";
        gallery_images[i] = imread(path_to_gallery_images+gallery_image_name);
        if(gallery_images[i].empty()){ cout << "Image "+to_string(i)+" empty. Ending program." << endl; return -1; }
        
        //Locate paintings
        
        //Recognise paintings
        
        //Overlay image names and locations on gallery images ---Shouldn't be in final program
        
    }
    
    //Compare recognised paintings to ground truths (calculate different metrics)
    
    
    return 0;
}

