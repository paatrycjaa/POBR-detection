/**
 * @file LogoDetection.h
 * @brief Header file for LogoDetection class, handling logo detection on image
 * 
 * @author Patrycja Cieplicka
 */

#ifndef LOGODETECTION_H
#define LOGODETECTION_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

//constatant values
const cv::Scalar MIN_YELLOW = cv::Scalar(20, 25, 20);
const cv::Scalar MAX_YELLOW = cv::Scalar(40, 165, 255);
const cv::Scalar MIN_WHITE = cv::Scalar(0, 165, 0);
const cv::Scalar MAX_WHITE = cv::Scalar(180, 255, 255);
const cv::Scalar MIN_LIGHTGREEN = cv::Scalar(35, 25, 40);
const cv::Scalar MAX_LIGHTGREEN = cv::Scalar(65, 170, 255);
const cv::Scalar MIN_GREEN = cv::Scalar(65, 25, 40);
const cv::Scalar MAX_GREEN = cv::Scalar(90, 165, 255);

const double MIN_M1_g = 0.265; //0.302811; 0.29757 0.358553M7
const double MAX_M1_g = 0.361; 
const double MIN_M7_g = 0.0172;     //0.0223432 0.021986 0.0321004
const double MAX_M7_g = 0.0322; 
const double MIN_W3_g = 1.3;//3.24993 1.92 4.02458
const double MAX_W3_g = 4.2;

const double MIN_M1_wy = 0.14; //0.166659; 0.16022
const double MAX_M1_wy = 0.206; 
const double MIN_M7_wy = 0.006;   //0.00675964; 0.006346
const double MAX_M7_wy = 0.0075; 
const double MIN_W3_wy = 0.02; //1.41793 ; 0.0224056
const double MAX_W3_wy = 2.93;


struct Point{
    Point() {};
    Point(int x, int y) : x_(x), y_(y) {};
    bool operator==(const Point& p){
        if (p.x_ == x_ && p.y_ == y_) return true;
        else return false;
    }
    int x_;
    int y_;
};

struct Segment{
    Segment(){};
    Segment(cv::Scalar color): color_(color){};
    std::vector<Point> points_; 
    cv::Scalar color_; 
};

class LogoDetection {
public:
    LogoDetection(){};
    cv::Mat thresholdYellowWhite(const cv::Mat& I);
    cv::Mat thresholdGreen(const cv::Mat& I);
    void recognize(cv::Mat& image1);
    bool analyzeSegmentGreen(const cv::Mat& I, const Segment &seg);
    bool analyzeSegmentWY(const cv::Mat& I, const Segment &seg);
    cv::Mat getSegmentatedImageWY();
    cv::Mat getSegmentatedImageG();
    cv::Mat convolution(cv::Mat& I, cv::Mat& kernel);

private:
    cv::Mat hlsImage;
    cv::Mat WY, G;

    int countCircum(const cv::Mat& I, const cv::Scalar &value);
    int countArea(const cv::Mat& I, const cv::Scalar &value);
    double W3(int area, int circumference);
    double momentPQ(const cv::Mat& I, int p, int q, const cv::Scalar &value);
    double M20(const cv::Mat& I, const cv::Scalar &value);
    double M02(const cv::Mat& I, const cv::Scalar &value);
    double M11(const cv::Mat& I, const cv::Scalar &value);
    double invariantM7(const cv::Mat& I, const cv::Scalar &value);
    double invariantM1(const cv::Mat& I, const cv::Scalar &value);
    Segment segmentation(cv::Mat& I, const cv::Scalar &color, Point startPoint);
    std::vector<Segment> performSegmentation(cv::Mat& I);

    int cut(int value);

    bool isYellow(int h, int l, int s);
    bool isWhite(int h, int l, int s);
    bool isGreenLight(int h, int l, int s);
    bool isGreen(int h, int l, int s);

    cv::Scalar getBGRColor(cv::RNG random);
    bool ifValidPosition(const cv::Mat& I, const Point &p);
    void uncolorSegment(cv::Mat& I, const std::vector<Segment> &seg);
    void drawRectangle (cv::Mat& I, const Segment &seg);
    void detectifInSegment (cv::Mat& I, const Segment &seg1, const Segment &seg2);

    //cv::Scalar getBGRColor();
    //std::vector<cv::Scalar> used_colors;
};


#endif	// LOGODETECTION_H