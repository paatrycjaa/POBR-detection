#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

const cv::Scalar MIN_YELLOW = cv::Scalar(20, 60, 20);
const cv::Scalar MAX_YELLOW = cv::Scalar(40, 200, 255);
const cv::Scalar MIN_WHITE = cv::Scalar(0, 165, 0);
const cv::Scalar MAX_WHITE = cv::Scalar(180, 255, 255);
const cv::Scalar MIN_LIGHTGREEN = cv::Scalar(35, 25, 40);
const cv::Scalar MAX_LIGHTGREEN = cv::Scalar(90, 200, 255);
const cv::Scalar MIN_GREEN = cv::Scalar(65, 50, 75);
const cv::Scalar MAX_GREEN = cv::Scalar(80, 150, 255);

const double MIN_M1_g = 0.25; //0.302811; 0.29757 0.358553M7
const double MAX_M1_g = 0.36; 
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

int countCircum(cv::Mat& I, cv::Scalar value){
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat C(I.rows,I.cols, CV_8UC3);
    cv::Mat I_(I.rows,I.cols, CV_8UC1);
    cv::Mat_<cv::Vec3b> L_;
    if (I.type()==CV_8UC1) {
        I_ = I;
    } else {
        L_ = I;
    }
    cv::Mat_<cv::Vec3b> circum = C;

    int circumference = 0;

    int dx[8] = {1,1,0,0,1,-1,-1,-1};
    int dy[8] = {0,1,1,-1,-1,1,0,-1};

    switch(I.channels())  {
    case 1:
        for( int i = 1; i < I.rows-1; ++i)
            for( int j = 1; j < I.cols-1; ++j )
                if (I_.at<uchar>(i,j) < 2){
                    for( int k = 0; k <3; k = k+2){
                        if(I_.at<uchar>(i-1+k,j) == 255 || I_.at<uchar>(i,j-1+k) == 255) {
                            circumference++;
                            circum(i,j)[0] = 255;
                            break;
                        }
                    }
                }
        break;
    case 3:
        for( int i = 1; i < L_.rows-1; ++i)
            for( int j = 1; j < L_.cols-1; ++j ){
                if (L_(i,j)[0] == value[0] && L_(i,j)[1] == value[1] && L_(i,j)[2] == value[2]) {

                    for(int k = 0; k <8; ++k){
                        if(L_(i+dx[k],j+dy[k])[0] != value[0] && L_(i+dx[k],j+dy[k])[1] != value[1] && L_(i+dx[k],j+dy[k])[2] != value[2]) {
                            circumference++;
                            circum(i,j)[0] = 255;
                            break;
                        }
                    }
                    
                }
            }
        break;
    default: break;    
    }
    //cv::imshow("elipsa_obwod",circum);
    //std::cout << circum.isContinuous() << std::endl;
    return circumference;

}

int countArea(cv::Mat& I, cv::Scalar value){
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat C(I.rows,I.cols, CV_8UC3);
    cv::Mat I_(I.rows,I.cols, CV_8UC1);
    cv::Mat_<cv::Vec3b> L_;
    if (I.type()==CV_8UC1) {
        I_ = I;
    } else {
        L_ = I;
    }

    cv::Mat_<cv::Vec3b> fig = C;

    int area = 0;
    switch(I.channels())  {
    case 1:
        for( int i = 1; i < I_.rows; ++i)
            for( int j = 1; j < I_.cols; ++j )
            if (I_.at<uchar>(i,j) == 0){
                area++;
                fig(i,j)[1] = 255;
            }
        break;
    case 3:
        for( int i = 1; i < I_.rows; ++i)
            for( int j = 1; j < I_.cols; ++j )
                if(L_(i,j)[0] == value[0] && L_(i,j)[1] == value[1] && L_(i,j)[2] == value[2]){
                    area++;
                    fig(i,j)[1] = 255;
                }
        break;
    default: break;
    }
    //cv::imshow("pole",fig);
    //std::cout << fig.isContinuous() << std::endl;
    return area;
}

double W3(int area, int circumference){
    double w3 = (circumference/(2*sqrt(M_PI * area))) - 1;
    return w3;
}

double momentPQ(cv::Mat& I, int p, int q, cv::Scalar value){
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat I_(I.rows,I.cols, CV_8UC1);
    cv::Mat_<cv::Vec3b> L_;
    if (I.type()==CV_8UC1) {
        I_ = I;
    } else {
        L_ = I;
    }

    double mpq = 0;

    switch(I.channels())  {
    case 1:
        for( int i = 1; i < I_.rows; ++i)
            for( int j = 1; j < I_.cols; ++j ){
                if (I_.at<uchar>(i,j) == 0) mpq += pow(i, p) * pow(j, q);
            }
        break;
    case 3:
        for( int i = 1; i < L_.rows; ++i)
            for( int j = 1; j < L_.cols; ++j ){
                if (L_(i,j)[0] == value[0] && L_(i,j)[1] == value[1] && L_(i,j)[2] == value[2]) mpq += pow(i, p) * pow(j, q);
            }
        break;
    default: break;
    }

    return mpq;
}

double M20(cv::Mat& I, cv::Scalar value){
    return momentPQ(I,2,0,value) - momentPQ(I,1,0,value)*momentPQ(I,1,0,value) / momentPQ(I,0,0,value);
}

double M02(cv::Mat& I, cv::Scalar value){
    return momentPQ(I,0,2,value) - momentPQ(I,0,1,value)*momentPQ(I,0,1,value) / momentPQ(I,0,0,value);
}

double M11(cv::Mat& I, cv::Scalar value){

    return momentPQ(I,1,1,value) - momentPQ(I,1,0,value)*momentPQ(I,0,1,value) / momentPQ(I,0,0,value);
}

double invariantM7(cv::Mat& I, cv::Scalar value){
    return  ( M20(I,value)*M02(I,value) - M11(I,value)*M11(I,value) )/ ( momentPQ(I,0,0,value)*momentPQ(I,0,0,value)*momentPQ(I,0,0,value)*momentPQ(I,0,0,value) );
}

double invariantM1(cv::Mat& I, cv::Scalar value){
    return (M20(I,value) + M02(I,value))/ (momentPQ(I,0,0,value)*momentPQ(I,0,0,value));
}

double countDegrees(cv::Mat& I, cv::Scalar value){
    cv::Mat_<cv::Vec3b> I_ = I;

    //centrum obrazu
    double i_ = momentPQ(I,1,0,value) / momentPQ(I,0,0,value); 
    double j_ = momentPQ(I,0,1,value) / momentPQ(I,0,0,value);
    //srodek geometryczny
    int minx =I_.rows, miny =I_.cols, maxx = 0, maxy = 0;

    for( int i = 1; i < I_.rows; ++i)
        for( int j = 1; j < I_.cols; ++j ){
            if(I_(i,j)[0] == value[0] && I_(i,j)[1] == value[1] && I_(i,j)[2] == value[2]){
                if(i > maxx) {maxx = i;}
                if(i < minx) {minx = i;}
                if(j > maxy) {maxy = j;}
                if(j < miny) {miny = j;}
            }
        }
    
    double x_ = (maxx + minx)/2; 
    double y_ = (maxy + miny)/2; 

    for(int i = 0; i < 3; ++i){
        I_(x_,y_)[i] = 0;
        I_(i_,j_)[i] = 0;
    }
    //std::cout << i_ <<"," << j_ <<","<< x_<<"," << y_ << std::endl;
    //cv::imshow("srodek",I_);
    //std::cout << I_.isContinuous() << std::endl;

    return (atan2(x_-i_, y_-j_) * double(180) )/ M_PI;
}

int cut(int value){
    if (value >= 255)
        return 255;
    else if (value <= 0)
        return 0;
    else
        return value;
}

cv::Mat convolution(cv::Mat& I, cv::Mat& kernel){
    CV_Assert(I.depth() != sizeof(uchar));
    //CV_Assert(kernel.depth() != sizeof(uchar));
    cv::Mat  conv(I.rows -2 ,I.cols -2, CV_8UC3);
    switch(I.channels())  {
    case 3:
        cv::Mat_<cv::Vec3b> _I = I;
        cv::Mat_<cv::Vec3b> _R = conv;
        double sum_G = 0, sum_B = 0, sum_R = 0;
        for(int i = 1; i < I.rows -1 ;++i){
            for(int j = 1; j < I.cols -1; ++j){
                for(int k = 0; k < kernel.rows; ++k){
                    for(int l = 0; l < kernel.cols; ++l){
                        sum_G += _I(i-1+k, j-1+l)[0] * kernel.at<double>(k,l) ;
                        sum_B += _I(i-1+k, j-1+l)[1] * kernel.at<double>(k,l) ;
                        sum_R += _I(i-1+k, j-1+l)[2] * kernel.at<double>(k,l) ;
                    }
                }
                _R(i-1,j-1)[0] = cut(sum_G);
                _R(i-1,j-1)[1] = cut(sum_B);
                _R(i-1,j-1)[2] = cut(sum_R);
                sum_G = 0; sum_B = 0; sum_R = 0;
            }
        }
        conv = _R;

    }
    return conv;
}


bool isYellow(int h, int l, int s){
   // std::cout << "sprawdzam zolty" << std::endl;
	if( (h<=MAX_YELLOW[0] && h>=MIN_YELLOW[0]) && (l<=MAX_YELLOW[1] && l>=MIN_YELLOW[1]) && (s<=MAX_YELLOW[2] && s>=MIN_YELLOW[2]) )
		return true;
	else
		return false;
}

bool isWhite(int h, int l, int s){
   // std::cout << "sprawdzam bialy" << std::endl;
	if( (h<=MAX_WHITE[0] && h>=MIN_WHITE[0]) && (l<=MAX_WHITE[1] && l>=MIN_WHITE[1]) && (s<=MAX_WHITE[2] && s>=MIN_WHITE[2]) )
		return true;
	else
		return false;
}

bool isGreenLight(int h, int l, int s){
    //std::cout << "sprawdzam jasno_zielony" << std::endl;
	if( (h<=MAX_LIGHTGREEN[0] && h>=MIN_LIGHTGREEN[0]) && (l<=MAX_LIGHTGREEN[1] && l>=MIN_LIGHTGREEN[1]) && (s<=MAX_LIGHTGREEN[2] && s>=MIN_LIGHTGREEN[2]) )
		return true;
	else
		return false;
}

bool isGreen(int h, int l, int s){
   // std::cout << "zielony" << std::endl;
	if( (h<=MAX_GREEN[0] && h>=MIN_GREEN[0]) && (l<=MAX_GREEN[1] && l>=MIN_GREEN[1]) && (s<=MAX_GREEN[2] && s>=MIN_GREEN[2]) )
		return true;
	else
		return false;
}


cv::Mat thresholdYellowWhite(cv::Mat& I, cv::Scalar MINcolor1, cv::Scalar Maxcolor1, cv::Scalar MINcolor2, cv::Scalar Maxcolor2){
    CV_Assert(I.depth() != sizeof(uchar));

    cv::Mat_<cv::Vec3b> _I = I;
	cv::Mat T (I.size(), CV_8UC3);
    cv::Mat R(I.rows ,I.cols, CV_8UC3);

   /* std::cout << I.rows << I.cols << std::endl;
    int i = 175, j = 350;
    std::cout << int(_I(i,j)[0]) << "," << int(_I(i,j)[1]) << "," << int(_I(i,j)[2]) << std::endl;

    cv::Mat_<cv::Vec3b> _temp = I;
    _temp(i,j)[0] = 0 ;
    _temp(i,j)[1] = 0 ;
    _temp(i,j)[2] = 0 ;
    I = _temp;*/


   for( int i = 1; i < _I.rows; ++i){
        for( int j = 1; j < _I.cols; ++j ){
            //std::cout << "jestem" << std::endl;
            if(isYellow(_I(i,j)[0], _I(i,j)[1], _I(i,j)[2]) || isWhite(_I(i,j)[0], _I(i,j)[1], _I(i,j)[2]) ){
                T.at<cv::Vec3b>(i,j)[0] = 255; T.at<cv::Vec3b>(i,j)[1] = 255; T.at<cv::Vec3b>(i,j)[2] = 255;
            }
            else{
                T.at<cv::Vec3b>(i,j)[0] = 0; T.at<cv::Vec3b>(i,j)[1] = 0; T.at<cv::Vec3b>(i,j)[2] = 0;
            }
        } 
    } 

    R = T;
    return R;
}

cv::Mat thresholdGreen(cv::Mat& I, cv::Scalar MINcolor1, cv::Scalar Maxcolor1, cv::Scalar MINcolor2, cv::Scalar Maxcolor2){
    CV_Assert(I.depth() != sizeof(uchar));

    cv::Mat_<cv::Vec3b> _I = I;
	cv::Mat T (I.size(), CV_8UC3);
    cv::Mat R(I.rows ,I.cols, CV_8UC3);

   for( int i = 1; i < _I.rows; ++i){
        for( int j = 1; j < _I.cols; ++j ){
            //std::cout << "jestem" << std::endl;
            if(isGreenLight(_I(i,j)[0], _I(i,j)[1], _I(i,j)[2]) || isGreen(_I(i,j)[0], _I(i,j)[1], _I(i,j)[2]) ){
                T.at<cv::Vec3b>(i,j)[0] = 255; T.at<cv::Vec3b>(i,j)[1] = 255; T.at<cv::Vec3b>(i,j)[2] = 255;
            }
            else{
                T.at<cv::Vec3b>(i,j)[0] = 0; T.at<cv::Vec3b>(i,j)[1] = 0; T.at<cv::Vec3b>(i,j)[2] = 0;
            }
        } 
    } 

    R = T;
    return R;
}

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

cv::Scalar getRGBColor(){
    //std::cout << "kolor" << std::endl;
    cv::Scalar color;
    for (int i = 0; i < 3; ++i){
        color[i] = rand()%256;
    }

    return color;
}

bool ifValidPosition(cv::Mat& I, Point p){
    //std::cout << "valid position" << std::endl;
    if (p.x_ < I.rows && p.x_ >= 0 && p.y_ < I.cols && p.y_ >= 0) return true;
    return false;
}

Segment segmentation(cv::Mat& I, cv::Scalar color, Point startPoint){
    //flood Fill algorithm - four points to check
    //std::cout << "segementacja" << std::endl;
    std::vector<Point> pointsQueue;
    Point temp, neigbour;
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    Segment segment = Segment(color);

    pointsQueue.push_back(startPoint);

    while(!pointsQueue.empty()){
        // take first point from queue
        temp = pointsQueue.front();
        // delete this point from queue as it is checked
        pointsQueue.erase(pointsQueue.begin());

        // color Point
        for(int i = 0; i < 3; ++i){
             I.at<cv::Vec3b>(temp.x_, temp.y_)[i] = color[i];
        }
        //add Point to segment
        segment.points_.push_back(temp);

        //check all neighbours if they are in segment
        for(int i = 0; i < 4; ++i){
            neigbour = Point(temp.x_ + dx[i], temp.y_ + dy[i]);
            if( ifValidPosition(I, neigbour) && I.at<cv::Vec3b>(neigbour.x_, neigbour.y_)[0] == 255 && I.at<cv::Vec3b>(neigbour.x_, neigbour.y_)[1] == 255
                                            && I.at<cv::Vec3b>(neigbour.x_, neigbour.y_)[2] == 255){
                if (std::find(pointsQueue.begin(), pointsQueue.end(), neigbour) == pointsQueue.end()){
                    pointsQueue.push_back(neigbour);
                }
            }
        }
    }
    return segment;
}

void uncolorSegment(cv::Mat& I, const Segment &seg){
   // std::cout << "uncolor" << std::endl;
    for (auto& point : seg.points_){
        for(int i = 0; i < 3; ++i){
             I.at<cv::Vec3b>(point.x_, point.y_)[i] = 0;
        }
    }
}

std::vector<Segment> performSegmentation(cv::Mat& I){
    std::vector<Segment> segments;
    cv::Scalar color;
    Segment temp;

    //std::cout << "perform segementacja" << std::endl;

    for(int i = 0; i < I.rows; ++i){
        for(int j = 0; j < I.cols; ++j){
            //std::cout << i << ","<< j << std::endl;
            if (I.at<cv::Vec3b>(i, j)[0] == 255 && I.at<cv::Vec3b>(i, j)[1] == 255 && I.at<cv::Vec3b>(i, j)[2] == 255){
                //std::cout << "sprawdzony_warunek" << std::endl;
                color = getRGBColor();
                temp = segmentation(I, color, Point(i,j));
                if (temp.points_.size() < 200) {
                    uncolorSegment(I, temp);
                }
                else{
                    segments.push_back(temp);
                }
            }
        }
    }

    std::cout << segments.size() << std::endl;

    return segments;

}

void drawRectangle (cv::Mat& I, Segment seg){
    int minx =I.rows, miny =I.cols, maxx = 0, maxy = 0;

    for( int i = 0; i < seg.points_.size(); ++i){
        if(seg.points_.at(i).x_ > maxx) {maxx = seg.points_.at(i).x_;}
        if(seg.points_.at(i).x_ < minx) {minx = seg.points_.at(i).x_;}
        if(seg.points_.at(i).y_ > maxy) {maxy = seg.points_.at(i).y_;}
        if(seg.points_.at(i).y_ < miny) {miny = seg.points_.at(i).y_;}
    }

    //std::cout << minx << "," << maxx << "," << miny << "," << maxy << std::endl;

    cv::rectangle(I, cv::Point(miny, minx), cv::Point(maxy, maxx), cv::Scalar(255, 0, 0), 2 );
}

bool checkifInSegment (cv::Mat& I, Segment seg1, Segment seg2){

    int minx =I.rows, miny =I.cols, maxx = 0, maxy = 0;

    for( int i = 0; i < seg1.points_.size(); ++i){
        if(seg1.points_.at(i).x_ > maxx) {maxx = seg1.points_.at(i).x_;}
        if(seg1.points_.at(i).x_ < minx) {minx = seg1.points_.at(i).x_;}
        if(seg1.points_.at(i).y_ > maxy) {maxy = seg1.points_.at(i).y_;}
        if(seg1.points_.at(i).y_ < miny) {miny = seg1.points_.at(i).y_;}
    }

    int minx2 =I.rows, miny2 =I.cols, maxx2 = 0, maxy2 = 0;

    for( int i = 0; i < seg2.points_.size(); ++i){
        if(seg2.points_.at(i).x_ > maxx2) {maxx2 = seg2.points_.at(i).x_;}
        if(seg2.points_.at(i).x_ < minx2) {minx2 = seg2.points_.at(i).x_;}
        if(seg2.points_.at(i).y_ > maxy2) {maxy2 = seg2.points_.at(i).y_;}
        if(seg2.points_.at(i).y_ < miny2) {miny2 = seg2.points_.at(i).y_;}
    }

    //drawRectangle(I, seg2);
    //std::cout << "Seg1" << std::endl;
    //std::cout << minx << "," << maxx << "," << miny << "," << maxy << std::endl;
    //std::cout << "Seg2" << std::endl;
    //std::cout << minx2 << "," << maxx2 << "," << miny2 << "," << maxy2 << std::endl;

    if (maxx < maxx2 && minx > minx2 && maxy < maxy2 && miny > miny2){
        drawRectangle(I, seg2);
    }

}

bool analyzeSegmentGreen(cv::Mat& I, Segment seg){
    int area = seg.points_.size();
    int circumference = countCircum(I, seg.color_);
    double M1 = invariantM1(I, seg.color_);
    double M7 = invariantM7(I, seg.color_);
    double w3 = W3(area, circumference);

    //std::cout << "area:" << area <<  "circum:" << circumference <<"M1:" << M1 << "M7:" << M7 << "w3:" << w3 << std::endl;
    if(M1 > MIN_M1_g && M1 < MAX_M1_g && M7 > MIN_M7_g && M7 < MAX_M7_g && w3 > MIN_W3_g && w3 < MAX_W3_g) return true;
    else return false;
}

bool analyzeSegmentWY(cv::Mat& I, Segment seg){
    int area = seg.points_.size();
    int circumference = countCircum(I, seg.color_);
    double M1 = invariantM1(I, seg.color_);
    double M7 = invariantM7(I, seg.color_);
    double w3 = W3(area, circumference);

    //std::cout << "area:" << area <<  "circum:" << circumference <<"M1:" << M1 << "M7:" << M7 << "w3:" << w3 << std::endl;
    if(M1 > MIN_M1_wy && M1 < MAX_M1_wy && M7 > MIN_M7_wy && M7 < MAX_M7_wy && w3 > MIN_W3_wy && w3 < MAX_W3_wy) {return true;}
    else return false;
}

void recognize(cv::Mat& image1){
    cv::Mat hlsImage;
	cv::cvtColor(image1, hlsImage, CV_BGR2HLS);
    cv::Mat WY, G;

	WY = thresholdYellowWhite(hlsImage, MIN_WHITE, MAX_WHITE, MIN_YELLOW, MAX_YELLOW);
    G = thresholdGreen(hlsImage, MIN_WHITE, MAX_WHITE, MIN_YELLOW, MAX_YELLOW);

    std::vector<Segment> seg_wy = performSegmentation(WY);
    std::vector<Segment> seg_g = performSegmentation(G);

    std::vector<Segment> seg_probably;

    for (auto &seg_1 : seg_wy){
        if(analyzeSegmentWY(WY, seg_1)) {
            seg_probably.push_back(seg_1);
            //drawRectangle(image1, seg_1);
        }
    }

    for (auto &seg_2 : seg_g){
        if(analyzeSegmentGreen(G, seg_2)) {
            seg_probably.push_back(seg_2);
            //drawRectangle(image1, seg_2);
        }
    }

    for(auto& seg : seg_probably){
        for(auto& segtwo : seg_probably) {
            checkifInSegment(image1,seg, segtwo);
        }
    }
}



int main(int, char *[]) {
    std::cout << "Start ..." << std::endl;
    srand ( time ( NULL ) ); 

    cv::Mat image1 = cv::imread("data/bp-1.jpg");
    cv::Mat image2 = cv::imread("data/bp-2.jpg");
    cv::Mat image3 = cv::imread("data/bp-3.jpg");
    cv::Mat image4 = cv::imread("data/bp-4.jpg");

    cv::Mat kernel_1 = (cv::Mat_<double>(3,3) << 1,1,1,
                                            1,12,1,
                                            1,1,1)/25;
    
    cv::Mat kernel_2 = (cv::Mat_<double>(3,3) << 1,-2,1,
                                            -2,5,-2,
                                            1,-2,1);



    cv::Mat inter = convolution(image1, kernel_1);
    cv::Mat contrast = convolution(image1, kernel_2);
    cv::Mat tog = convolution(inter, kernel_2);

    //cv::imshow("kon",contrast);
    //cv::imshow("inter",inter);
    //cv::imshow("together",tog);

    recognize(image1);
    recognize(image2);
    recognize(image3);
    recognize(image4);


    cv::imshow("bp-1",image1);
    cv::imshow("bp-2",image2);
    cv::imshow("bp-3",image3);
    cv::imshow("bp-4",image4);
   // cv::imshow("bp-2-wy",WY);
    //cv::imshow("bp-2-g",G);

    cv::waitKey(-1);
    return 0;
}
