#ifndef IPM_COMPOSER_H
#define IPM_COMPOSER_H

#include <opencv2/opencv.hpp>
#include "viwo_utils.h"

enum class SvcIndex_t : int{
    VOID = -1,
    FRONT = 0,
    LEFT = 1,
    REAR = 2,
    RIGHT = 3,
    MAX = 4,
    ALL = 5,
};

typedef struct{
    double time;
    cv::Mat img_front;
    cv::Mat img_left;
    cv::Mat img_rear;
    cv::Mat img_right;
    cv::Mat img_ipm;
} SvcPairedImages_t;

class IpmComposer
{
public:
    static constexpr float DT_THRESHOLD_SEC_ = 0.02f;
    static constexpr int BUFFER_MAX_ = 10;
    static constexpr int BEV_H_ = 640;
    static constexpr int BEV_W_ = 640;
    static constexpr float BEV_XMAX_ = 16.0;
    static constexpr float BEV_YMAX_ = 16.0;
    /* define center point of cameras, 1280*1280, 32*32m */
    // static constexpr float SVC_FRONT_X0 = 640.0f;
    // static constexpr float SVC_FRONT_Y0 = 536.8f;
    // static constexpr float SVC_LEFT_X0 = 596.0f;
    // static constexpr float SVC_LEFT_Y0 = 636.8f;
    // static constexpr float SVC_REAR_X0 = 640.0f;
    // static constexpr float SVC_REAR_Y0 = 744.8f;
    // static constexpr float SVC_RIGHT_X0 = 684.0f;
    // static constexpr float SVC_RIGHT_Y0 = 636.8f;

    /* define center point of cameras, 1080*360, 54*18m */
    // static constexpr float SVC_FRONT_X0 = 180.0f;
    // static constexpr float SVC_FRONT_Y0 = 488.4f;
    // static constexpr float SVC_LEFT_X0 = 158.0f;
    // static constexpr float SVC_LEFT_Y0 = 538.4f;
    // static constexpr float SVC_REAR_X0 = 180.0f;
    // static constexpr float SVC_REAR_Y0 = 592.4f;
    // static constexpr float SVC_RIGHT_X0 = 202.0f;
    // static constexpr float SVC_RIGHT_Y0 = 538.4f;

    /* define center point of cameras, 640*640, 16*16m */
    static constexpr float SVC_FRONT_X0 = 320.0f;
    static constexpr float SVC_FRONT_Y0 = 216.8f;
    static constexpr float SVC_LEFT_X0 = 276.0f;
    static constexpr float SVC_LEFT_Y0 = 316.8f;
    static constexpr float SVC_REAR_X0 = 320.0f;
    static constexpr float SVC_REAR_Y0 = 424.8f;
    static constexpr float SVC_RIGHT_X0 = 364.0f;
    static constexpr float SVC_RIGHT_Y0 = 316.8f;

private:
    cv::Mat homo_svc_front_;
    cv::Mat homo_svc_left_;
    cv::Mat homo_svc_rear_;
    cv::Mat homo_svc_right_;
    cv::Mat ipm_mask_svc_front_;
    cv::Mat ipm_mask_svc_left_;
    cv::Mat ipm_mask_svc_rear_;
    cv::Mat ipm_mask_svc_right_;

public:
    IpmComposer() {
        homo_svc_front_ = cv::Mat::eye(3, 3, CV_32F);
        homo_svc_left_ = cv::Mat::eye(3, 3, CV_32F);
        homo_svc_rear_ = cv::Mat::eye(3, 3, CV_32F);
        homo_svc_right_ = cv::Mat::eye(3, 3, CV_32F);

        ipm_mask_svc_front_ = cv::Mat::ones(BEV_H_, BEV_W_, CV_8UC1);
        ipm_mask_svc_left_ = cv::Mat::ones(BEV_H_, BEV_W_, CV_8UC1);
        ipm_mask_svc_rear_ = cv::Mat::ones(BEV_H_, BEV_W_, CV_8UC1);
        ipm_mask_svc_right_ = cv::Mat::ones(BEV_H_, BEV_W_, CV_8UC1);
    }

    ~IpmComposer() {}

    void SetHomography(cv::Mat& homo_svc_front, cv::Mat& homo_svc_left, cv::Mat& homo_svc_rear, cv::Mat& homo_svc_right){
        homo_svc_front_ = homo_svc_front;
        fprintf(stderr, "homo_svc_front_: %s\n", ViwoUtils::CvMat2Str(homo_svc_front_).c_str());
        homo_svc_left_ = homo_svc_left;
        fprintf(stderr, "homo_svc_left_: %s\n", ViwoUtils::CvMat2Str(homo_svc_left_).c_str());
        homo_svc_rear_ = homo_svc_rear;
        fprintf(stderr, "homo_svc_rear_: %s\n", ViwoUtils::CvMat2Str(homo_svc_rear_).c_str());
        homo_svc_right_ = homo_svc_right;
        fprintf(stderr, "homo_svc_right_: %s\n", ViwoUtils::CvMat2Str(homo_svc_right_).c_str());

        GenIpmMasksBow1();
    }

    void Compose(SvcPairedImages_t& pis, bool debug_save=false, std::string path="./"){
        HANG_STOPWATCH();
        const cv::Mat& img_front = pis.img_front;
        const cv::Mat& img_left = pis.img_left;
        const cv::Mat& img_rear = pis.img_rear;
        const cv::Mat& img_right = pis.img_right;
        cv::Mat& ipm = pis.img_ipm;

        ipm = cv::Mat::zeros(BEV_H_, BEV_W_, CV_8UC3);
        cv::Mat tmp(BEV_H_, BEV_W_, CV_8UC3);
        cv::warpPerspective(img_front, tmp, homo_svc_front_, tmp.size(), cv::INTER_NEAREST);
        tmp.setTo(cv::Scalar(0,0,0), ipm_mask_svc_front_);
        ipm += tmp;

        cv::warpPerspective(img_left, tmp, homo_svc_left_, tmp.size(), cv::INTER_NEAREST);
        tmp.setTo(cv::Scalar(0,0,0), ipm_mask_svc_left_);
        ipm += tmp;

        cv::warpPerspective(img_rear, tmp, homo_svc_rear_, tmp.size(), cv::INTER_NEAREST);
        tmp.setTo(cv::Scalar(0,0,0), ipm_mask_svc_rear_);
        ipm += tmp;

        cv::warpPerspective(img_right, tmp, homo_svc_right_, tmp.size(), cv::INTER_NEAREST);
        tmp.setTo(cv::Scalar(0,0,0), ipm_mask_svc_right_);
        ipm += tmp;

        if(debug_save){
            std::string tstr = std::to_string(pis.time);
            std::string filepath = path + "/" + tstr + "_ipm.png";
            cv::imwrite(filepath.c_str(), ipm);
#if 0
            filepath = path + "/" + tstr + "_front.png";
            cv::imwrite(filepath.c_str(), img_front);

            filepath = path + "/" + tstr + "_left.png";
            cv::imwrite(filepath.c_str(), img_left);

            filepath = path + "/" + tstr + "_rear.png";
            cv::imwrite(filepath.c_str(), img_rear);

            filepath = path + "/" + tstr + "_right.png";
            cv::imwrite(filepath.c_str(), img_right);
#endif
        }
    }

private:
    /* 
      line_left_forward: svc_left_center, (0, R*h)
      line_right_forward: svc_right_center, (w, R*h)
      line_left_rear: svc_left_center, (0, (1-R)*h)
      line_right_rear: svc_right_center, (w, (1-R)*h)
      k = (y2-y1)/(x1-x2)
      b = (x2y1-x1y2)/(x1-x2)
     */
    void GenIpmMasksBow2(){
        //ratio of height start-point, 0.0~0.5
        const float R = 0.27f; 

        /* line_left_forward */
        float k_lf, b_lf;
        GetKBFromTwoPoints(SVC_LEFT_X0, SVC_LEFT_Y0, 0.0f, R*BEV_H_, k_lf, b_lf);
        /* line_right_forward */
        float k_rf, b_rf;
        GetKBFromTwoPoints(SVC_RIGHT_X0, SVC_RIGHT_Y0, BEV_W_, R*BEV_H_, k_rf, b_rf);
        /* line_left_rear */
        float k_lr, b_lr;
        GetKBFromTwoPoints(SVC_LEFT_X0, SVC_LEFT_Y0, 0, (1.0f-R)*BEV_H_, k_lr, b_lr);
        /* line_right_rear */
        float k_rr, b_rr;
        GetKBFromTwoPoints(SVC_RIGHT_X0, SVC_RIGHT_Y0, BEV_W_, (1.0f-R)*BEV_H_, k_rr, b_rr);

        for(int y=0; y<BEV_H_; y++){
            for(int x=0; x<BEV_W_; x++){
                float line_lf = k_lf*x + y + b_lf;
                float line_rf = k_rf*x + y + b_rf;
                float line_lr = k_lr*x + y + b_lr;
                float line_rr = k_rr*x + y + b_rr;
                /* svc_front area */
                if(line_lf<=0.0f && line_rf<0.0f && y<SVC_FRONT_Y0){
                    ipm_mask_svc_front_.at<uchar>(y, x) = 0;
                }
                /* svc_left area */
                else if(line_lf>0.0f && line_lr<=0.0f){
                    ipm_mask_svc_left_.at<uchar>(y, x) = 0;
                }
                /* svc_rear area */
                else if(line_lr>=0.0f && line_rr>0.0f && y>SVC_REAR_Y0){
                    ipm_mask_svc_rear_.at<uchar>(y, x) = 0;
                }
                /* svc_right area */
                else if(line_rr<0.0f && line_rf>=0.0f){
                    ipm_mask_svc_right_.at<uchar>(y, x) = 0;
                }
            }
        }
        // std::string filepath = "/home/hugoliu/github/catkin_ws/src/VIW-Fusion/output";
        // cv::imwrite((filepath+"/svc_front_mask.png").c_str(), 255*ipm_mask_svc_front_);
        // cv::imwrite((filepath+"/svc_left_mask.png").c_str(), 255*ipm_mask_svc_left_);
        // cv::imwrite((filepath+"/svc_rear_mask.png").c_str(), 255*ipm_mask_svc_rear_);
        // cv::imwrite((filepath+"/svc_right_mask.png").c_str(), 255*ipm_mask_svc_right_);
    }

    /* 
      line_left_forward: svc_left_center, (R*w, 0)
      line_right_forward: svc_right_center, ((1-R)*w, 0)
      line_left_rear: svc_left_center, (R*w, h)
      line_right_rear: svc_right_center, ((1-R)*w, h)
      k = (y2-y1)/(x1-x2)
      b = (x2y1-x1y2)/(x1-x2)
     */
    void GenIpmMasksBow1(){
        //ratio of width start-point, 0.0~0.5, 0.128 for w=640, 0.16 for w=1280
        const float R = 0.128f;

        /* line_left_forward */
        float k_lf, b_lf;
        GetKBFromTwoPoints(SVC_LEFT_X0, SVC_LEFT_Y0, R*BEV_W_, 0.0f, k_lf, b_lf);
        /* line_right_forward */
        float k_rf, b_rf;
        GetKBFromTwoPoints(SVC_RIGHT_X0, SVC_RIGHT_Y0, (1.0f-R)*BEV_W_, 0.0f, k_rf, b_rf);
        /* line_left_rear */
        float k_lr, b_lr;
        GetKBFromTwoPoints(SVC_LEFT_X0, SVC_LEFT_Y0, R*BEV_W_, BEV_H_, k_lr, b_lr);
        /* line_right_rear */
        float k_rr, b_rr;
        GetKBFromTwoPoints(SVC_RIGHT_X0, SVC_RIGHT_Y0, (1.0f-R)*BEV_W_, BEV_H_, k_rr, b_rr);

        for(int y=0; y<BEV_H_; y++){
            for(int x=0; x<BEV_W_; x++){
                float line_lf = k_lf*x + y + b_lf;
                float line_rf = k_rf*x + y + b_rf;
                float line_lr = k_lr*x + y + b_lr;
                float line_rr = k_rr*x + y + b_rr;
                /* svc_front area */
                if(line_lf<=0.0f && line_rf<0.0f && y<SVC_FRONT_Y0){
                    ipm_mask_svc_front_.at<uchar>(y, x) = 0;
                }
                /* svc_left area */
                else if(line_lf>0.0f && line_lr<=0.0f){
                    ipm_mask_svc_left_.at<uchar>(y, x) = 0;
                }
                /* svc_rear area */
                else if(line_lr>=0.0f && line_rr>0.0f && y>SVC_REAR_Y0){
                    ipm_mask_svc_rear_.at<uchar>(y, x) = 0;
                }
                /* svc_right area */
                else if(line_rr<0.0f && line_rf>=0.0f){
                    ipm_mask_svc_right_.at<uchar>(y, x) = 0;
                }
            }
        }
        // std::string filepath = "/home/hugoliu/github/catkin_ws/src/VIW-Fusion/output";
        // cv::imwrite((filepath+"/svc_front_mask.png").c_str(), 255*ipm_mask_svc_front_);
        // cv::imwrite((filepath+"/svc_left_mask.png").c_str(), 255*ipm_mask_svc_left_);
        // cv::imwrite((filepath+"/svc_rear_mask.png").c_str(), 255*ipm_mask_svc_rear_);
        // cv::imwrite((filepath+"/svc_right_mask.png").c_str(), 255*ipm_mask_svc_right_);
    }

    void GetKBFromTwoPoints(const float x1, const float y1, const float x2, const float y2, float& k, float& b){
        k = (y2-y1)/(x1-x2);
        b = (x2*y1-x1*y2)/(x1-x2);
    }

    /* 
      line1-(0, 0)-(w, h): -1*BEV_H_/BEV_W_*x + y = 0
      line2-(w, 0)-(0, h): BEV_H_/BEV_W_*x + y - BEV_H_ = 0
     */
    void GenIpmMasksOrtho(){
        const float k1 = -1*BEV_H_/(float)BEV_W_;
        const float b1 = 0.0f;
        const float k2 = BEV_H_/(float)BEV_W_;
        const float b2 = -1 * BEV_H_;

        for(int y=0; y<BEV_H_; y++){
            for(int x=0; x<BEV_W_; x++){
                float line1 = k1*x + y + b1;
                float line2 = k2*x + y + b2;
                /* svc_front area */
                if(line1<=0.0f && line2<0.0f){
                    ipm_mask_svc_front_.at<uchar>(y, x) = 0;
                }
                /* svc_left area */
                else if(line1>0.0f && line2<=0.0f){
                    ipm_mask_svc_left_.at<uchar>(y, x) = 0;
                }
                /* svc_rear area */
                else if(line1>=0.0f && line2>0.0f){
                    ipm_mask_svc_rear_.at<uchar>(y, x) = 0;
                }
                /* svc_right area */
                else if(line1<0.0f && line2>=0.0f){
                    ipm_mask_svc_right_.at<uchar>(y, x) = 0;
                }

            }
        }
        // std::string filepath = "/home/hugoliu/github/catkin_ws/src/VIW-Fusion/output";
        // cv::imwrite((filepath+"/svc_front_mask.png").c_str(), 255*ipm_mask_svc_front_);
        // cv::imwrite((filepath+"/svc_left_mask.png").c_str(), 255*ipm_mask_svc_left_);
        // cv::imwrite((filepath+"/svc_rear_mask.png").c_str(), 255*ipm_mask_svc_rear_);
        // cv::imwrite((filepath+"/svc_right_mask.png").c_str(), 255*ipm_mask_svc_right_);
    }

    /* only reserve positive half */
    void GenIpmMasksHalf(){
        for(int y=0; y<BEV_H_; y++){
            for(int x=0; x<BEV_W_; x++){
                if(x > BEV_W_/2){
                    ipm_mask_svc_right_.at<uchar>(y, x) = 0;
                }else{
                    ipm_mask_svc_left_.at<uchar>(y, x) = 0;
                }

                if(y > BEV_H_/2){
                    ipm_mask_svc_rear_.at<uchar>(y, x) = 0;
                }else{
                    ipm_mask_svc_front_.at<uchar>(y, x) = 0;
                }
            }
        }
        // std::string filepath = "/home/hugoliu/github/catkin_ws/src/VIW-Fusion/output";
        // cv::imwrite((filepath+"/svc_front_mask.png").c_str(), 255*ipm_mask_svc_front_);
        // cv::imwrite((filepath+"/svc_left_mask.png").c_str(), 255*ipm_mask_svc_left_);
        // cv::imwrite((filepath+"/svc_rear_mask.png").c_str(), 255*ipm_mask_svc_rear_);
        // cv::imwrite((filepath+"/svc_right_mask.png").c_str(), 255*ipm_mask_svc_right_);
    }

};

#endif //IPM_COMPOSER_H