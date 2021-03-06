/**
 * @file LogoDetection.h
 * @brief Source file for LogoDetection class, handling logo detection on image
 * 
 * @author Patrycja Cieplicka
 */

#include "LogoDetection.h"


bool LogoDetection::isYellow(int h, int l, int s){
	if( (h<=MAX_YELLOW[0] && h>=MIN_YELLOW[0]) && (l<=MAX_YELLOW[1] && l>=MIN_YELLOW[1]) && (s<=MAX_YELLOW[2] && s>=MIN_YELLOW[2]) )
		return true;
	else
		return false;
}

bool LogoDetection::isWhite(int h, int l, int s){
	if( (h<=MAX_WHITE[0] && h>=MIN_WHITE[0]) && (l<=MAX_WHITE[1] && l>=MIN_WHITE[1]) && (s<=MAX_WHITE[2] && s>=MIN_WHITE[2]) )
		return true;
	else
		return false;
}

bool LogoDetection::isGreenLight(int h, int l, int s){
	if( (h<=MAX_LIGHTGREEN[0] && h>=MIN_LIGHTGREEN[0]) && (l<=MAX_LIGHTGREEN[1] && l>=MIN_LIGHTGREEN[1]) && (s<=MAX_LIGHTGREEN[2] && s>=MIN_LIGHTGREEN[2]) )
		return true;
	else
		return false;
}

bool LogoDetection::isGreen(int h, int l, int s){
	if( (h<=MAX_GREEN[0] && h>=MIN_GREEN[0]) && (l<=MAX_GREEN[1] && l>=MIN_GREEN[1]) && (s<=MAX_GREEN[2] && s>=MIN_GREEN[2]) )
		return true;
	else
		return false;
}


cv::Mat LogoDetection::thresholdYellowWhite(const cv::Mat& I){
    CV_Assert(I.depth() != sizeof(uchar));

    cv::Mat_<cv::Vec3b> _I = I;
	cv::Mat T (I.size(), CV_8UC3);
    cv::Mat R(I.rows ,I.cols, CV_8UC3);


   for( int i = 1; i < _I.rows; ++i){
        for( int j = 1; j < _I.cols; ++j ){
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

cv::Mat LogoDetection::thresholdGreen(const cv::Mat& I){
    CV_Assert(I.depth() != sizeof(uchar));

    cv::Mat_<cv::Vec3b> _I = I;
	cv::Mat T (I.size(), CV_8UC3);
    cv::Mat R(I.rows ,I.cols, CV_8UC3);

   for( int i = 1; i < _I.rows; ++i){
        for( int j = 1; j < _I.cols; ++j ){
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


cv::Scalar LogoDetection::getBGRColor(cv::RNG random){

    int color = (unsigned) random;
    return cv::Scalar( color&255, (color>>8)&255, (color>>16)&255 );

}

bool LogoDetection::ifValidPosition(const cv::Mat& I, const Point &p){
    if (p.x_ < I.rows && p.x_ >= 0 && p.y_ < I.cols && p.y_ >= 0) return true;
    return false;
}

Segment LogoDetection::segmentation(cv::Mat& I, const cv::Scalar &color, Point startPoint){
    //Flood file algorithm to get every segment

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
            if( ifValidPosition(I, neigbour) && I.at<cv::Vec3b>(neigbour.x_, neigbour.y_)[0] == 255  
                                            && I.at<cv::Vec3b>(neigbour.x_, neigbour.y_)[1] == 255
                                            && I.at<cv::Vec3b>(neigbour.x_, neigbour.y_)[2] == 255){
                if (std::find(pointsQueue.begin(), pointsQueue.end(), neigbour) == pointsQueue.end()){
                    pointsQueue.push_back(neigbour);
                }
            }
        }
    }
    return segment;
}

void LogoDetection::uncolorSegment(cv::Mat& I, const std::vector<Segment> &seg){
    for (auto& segment : seg)
        for (auto& point : segment.points_){
            for(int i = 0; i < 3; ++i){
                I.at<cv::Vec3b>(point.x_, point.y_)[i] = 0;
            }
        }
}

std::vector<Segment> LogoDetection::performSegmentation(cv::Mat& I){
    std::vector<Segment> segments;
    std::vector<Segment> unsegments;
    cv::Scalar color;
    Segment temp;

    int n = 1;
    for(int i = 0; i < I.rows; ++i){
        for(int j = 0; j < I.cols; ++j){
            if (I.at<cv::Vec3b>(i, j)[0] == 255 && I.at<cv::Vec3b>(i, j)[1] == 255 && I.at<cv::Vec3b>(i, j)[2] == 255){
                cv::RNG random(n);
				color = getBGRColor(random);
                temp = segmentation(I, color, Point(i,j));
                if (temp.points_.size() < 200) {
                    unsegments.push_back(temp);
                    n++;
                }
                else{
                    segments.push_back(temp);
                    n++;
                }
            }
        }
    }
    uncolorSegment(I, unsegments); //uncolor segments which have less than 200 points

    return segments;

}

int LogoDetection::countCircum(const cv::Mat& I, const cv::Scalar &value){
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

    return circumference;

}

int LogoDetection::countArea(const cv::Mat& I, const cv::Scalar &value){
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
    return area;
}

double LogoDetection::W3(int area, int circumference){
    double w3 = (circumference/(2*sqrt(M_PI * area))) - 1;
    return w3;
}

double LogoDetection::momentPQ(const cv::Mat& I, int p, int q, const cv::Scalar &value){
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

double LogoDetection::M20(const cv::Mat& I, const cv::Scalar &value){
    return momentPQ(I,2,0,value) - momentPQ(I,1,0,value)*momentPQ(I,1,0,value) / momentPQ(I,0,0,value);
}

double LogoDetection::M02(const cv::Mat& I, const cv::Scalar &value){
    return momentPQ(I,0,2,value) - momentPQ(I,0,1,value)*momentPQ(I,0,1,value) / momentPQ(I,0,0,value);
}

double LogoDetection::M11(const cv::Mat& I, const cv::Scalar &value){

    return momentPQ(I,1,1,value) - momentPQ(I,1,0,value)*momentPQ(I,0,1,value) / momentPQ(I,0,0,value);
}

double LogoDetection::invariantM7(const cv::Mat& I, const cv::Scalar &value){
    return  ( M20(I,value)*M02(I,value) - M11(I,value)*M11(I,value) )/ ( momentPQ(I,0,0,value)*momentPQ(I,0,0,value)*momentPQ(I,0,0,value)*momentPQ(I,0,0,value) );
}

double LogoDetection::invariantM1(const cv::Mat& I, const cv::Scalar &value){
    return (M20(I,value) + M02(I,value))/ (momentPQ(I,0,0,value)*momentPQ(I,0,0,value));
}

void LogoDetection::drawRectangle (cv::Mat& I, const Segment &seg){
    int minx =I.rows, miny =I.cols, maxx = 0, maxy = 0;

    for( int i = 0; i < seg.points_.size(); ++i){
        if(seg.points_.at(i).x_ > maxx) {maxx = seg.points_.at(i).x_;}
        if(seg.points_.at(i).x_ < minx) {minx = seg.points_.at(i).x_;}
        if(seg.points_.at(i).y_ > maxy) {maxy = seg.points_.at(i).y_;}
        if(seg.points_.at(i).y_ < miny) {miny = seg.points_.at(i).y_;}
    }

    cv::rectangle(I, cv::Point(miny-2, minx-2), cv::Point(maxy+2, maxx+2), cv::Scalar(255, 0, 0));
}

void LogoDetection::detectifInSegment (cv::Mat& I, const Segment &seg1, const Segment &seg2){

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

    if (maxx < maxx2 && minx > minx2 && maxy < maxy2 && miny > miny2){
        drawRectangle(I, seg2);
    }

}

bool LogoDetection::analyzeSegmentGreen(const cv::Mat& I, const Segment &seg){
    int area = seg.points_.size();
    int circumference = countCircum(I, seg.color_);
    double M1 = invariantM1(I, seg.color_);
    double M7 = invariantM7(I, seg.color_);
    double w3 = W3(area, circumference);

    if(M1 > MIN_M1_g && M1 < MAX_M1_g && M7 > MIN_M7_g && M7 < MAX_M7_g && w3 > MIN_W3_g && w3 < MAX_W3_g) return true;
    else return false;
}

bool LogoDetection::analyzeSegmentWY(const cv::Mat& I, const Segment &seg){
    int area = seg.points_.size();
    int circumference = countCircum(I, seg.color_);
    double M1 = invariantM1(I, seg.color_);
    double M7 = invariantM7(I, seg.color_);
    double w3 = W3(area, circumference);

    if(M1 > MIN_M1_wy && M1 < MAX_M1_wy && M7 > MIN_M7_wy && M7 < MAX_M7_wy && w3 > MIN_W3_wy && w3 < MAX_W3_wy) {return true;}
    else return false;
}

void LogoDetection::recognize(cv::Mat& image1){

	cv::cvtColor(image1, hlsImage, CV_BGR2HLS);
    
	WY = thresholdYellowWhite(hlsImage);
    G = thresholdGreen(hlsImage);

    std::vector<Segment> seg_wy = performSegmentation(WY);
    std::vector<Segment> seg_g = performSegmentation(G);

    std::vector<Segment> seg_probably;

    for (auto &seg_1 : seg_wy){
        if(analyzeSegmentWY(WY, seg_1)) {
            seg_probably.push_back(seg_1);
        }
    }

    for (auto &seg_2 : seg_g){
        if(analyzeSegmentGreen(G, seg_2)) {
            seg_probably.push_back(seg_2);
        }
    }

    for(auto& seg : seg_probably){
        for(auto& segtwo : seg_probably) {
            detectifInSegment(image1,seg, segtwo);
        }
    }
}

int LogoDetection::cut(int value){
    if (value >= 255)
        return 255;
    else if (value <= 0)
        return 0;
    else
        return value;
}

cv::Mat LogoDetection::convolution(cv::Mat& I, cv::Mat& kernel){
    CV_Assert(I.depth() != sizeof(uchar));

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

cv::Mat LogoDetection::getSegmentatedImageWY(){
    return WY;
}
cv::Mat LogoDetection::getSegmentatedImageG(){
    return G;
}

/*cv::Scalar LogoDetection::getBGRColor(){
    int b,g,r;
    cv::Scalar color;
    bool used;

    do{
        b = rand() % 256;
        g = rand() % 256;
        r = rand() % 256;

        used = false;
        for(auto& color : used_colors){
            if(color[0] == b && color[1] == g && color[2] == r) {
                used = true;
                break;
            }
        }
    }while (used);
    
    color = cv::Scalar(b,g,r);
    used_colors.push_back(color);

    return color;
}*/
