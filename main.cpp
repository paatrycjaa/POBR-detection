/**
 * @file Main.cpp
 * @brief Source file for main - LogoDetection program, handling logo detection on image
 * 
 * @author Patrycja Cieplicka
 */

#include "LogoDetection.h"


int main(int, char *[]) {
    std::cout << "BP Logo detection" << std::endl;

    std::cout << "Read image" << std::endl;
    cv::Mat image1 = cv::imread("data/bp-1.jpg");
    cv::Mat image2 = cv::imread("data/bp-2.jpg");
    cv::Mat image3 = cv::imread("data/bp-3.jpg");
    cv::Mat image4 = cv::imread("data/bp-4.jpg");

    LogoDetection logo;

    std::cout << "Recognize image 1" << std::endl;
    logo.recognize(image1);
    cv::Mat image1_segG = logo.getSegmentatedImageG();
    cv::Mat image1_segWY = logo.getSegmentatedImageWY();
    std::cout << "Recognize image 2" << std::endl;
    logo.recognize(image2);
    cv::Mat image2_segG = logo.getSegmentatedImageG();
    cv::Mat image2_segWY = logo.getSegmentatedImageWY();
    std::cout << "Recognize image 3" << std::endl;
    logo.recognize(image3);
    cv::Mat image3_segG = logo.getSegmentatedImageG();
    cv::Mat image3_segWY = logo.getSegmentatedImageWY();
    std::cout << "Recognize image 4" << std::endl;
    logo.recognize(image4);
    cv::Mat image4_segG = logo.getSegmentatedImageG();
    cv::Mat image4_segWY = logo.getSegmentatedImageWY();


    cv::imshow("bp-1",image1);
    cv::imshow("bp-2",image2);
    cv::imshow("bp-3",image3);
    cv::imshow("bp-3g",image4_segG);
     cv::imshow("bp-3wy",image4_segWY);
    cv::imshow("bp-4",image4);

    cv::waitKey(-1);
    return 0;
}
