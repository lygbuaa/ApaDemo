#pragma once

#include <opencv2/opencv.hpp>

class CvParamLoader
{
public:
    cv::Mat homo_svc_front_, homo_svc_left_, homo_svc_rear_, homo_svc_right_;
    std::string project_rootdir_;
    std::string output_path_;
    std::string pcr_model_path_;
    std::string psd_model_path_;
    std::string dataset_path_;

private:
    std::string yaml_path_;
    cv::FileStorage fs_;

public:
    CvParamLoader(std::string yaml_path){
        yaml_path_ = yaml_path;
        fs_.open(yaml_path, cv::FileStorage::READ);
        if(!fs_.isOpened()){
            fprintf(stderr, "open config file failed: %s\n", yaml_path.c_str());
        }
        assert(fs_.isOpened());
        load_params();
    }

    ~CvParamLoader(){
        if(fs_.isOpened()){
            fs_.release();
        }
    }

    void load_params(){
        fs_["homo_svc_front"] >> homo_svc_front_;
        fs_["homo_svc_left"] >> homo_svc_left_;
        fs_["homo_svc_rear"] >> homo_svc_rear_;
        fs_["homo_svc_right"] >> homo_svc_right_;

        fs_["project_rootdir"] >> project_rootdir_;
        fs_["output_path"] >> output_path_;
        output_path_ = project_rootdir_ + output_path_;

        fs_["pcr_model_path"] >> pcr_model_path_;
        pcr_model_path_ = project_rootdir_ + pcr_model_path_;

        fs_["psd_model_path"] >> psd_model_path_;
        psd_model_path_ = project_rootdir_ + psd_model_path_;

        fs_["dataset_path"] >> dataset_path_;
        dataset_path_ = project_rootdir_ + dataset_path_;

        fprintf(stderr, "dataset_path_: %s, output_path_: %s, pcr_model_path_: %s, psd_model_path_: %s\n", \
            dataset_path_.c_str(), output_path_.c_str(), pcr_model_path_.c_str(), psd_model_path_.c_str());
    }
};