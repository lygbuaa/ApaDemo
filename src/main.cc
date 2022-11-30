#include <iostream>
#include <fstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include "PsdWrapper.h"
#include "JsonDataset.h"
#include "IpmComposer.h"
#include "CvParamLoader.h"

void signal_handler(int sig_num){
	std::cout << "\n@q@ --> it's quit signal: " << sig_num << ", see you later.\n";
	exit(sig_num);
}

bool is_cuda_avaliable(){
    std::cout << "opencv version: " << CV_VERSION << std::endl;
    int cnt = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "getCudaEnabledDeviceCount: " << cnt << std::endl;
    for(int i=0; i<cnt; ++i){
        cv::cuda::printCudaDeviceInfo(i);
    }
    if(cnt > 0){
        std::cout << "current cuda device: " << cv::cuda::getDevice() << std::endl;
    }
    return (cnt > 0);
}

void do_work(){
    const std::string config_file = "/mnt/c/work/github/ApaDemo/config/ipm_carla_town04.yaml";
    CvParamLoader cpl(config_file);

    std::unique_ptr<IpmComposer> ptr_ipm_composer(new IpmComposer());
    ptr_ipm_composer->SetHomography(cpl.homo_svc_front_, cpl.homo_svc_left_, cpl.homo_svc_rear_, cpl.homo_svc_right_);

    std::unique_ptr<psdonnx::PsdWrapper> ptr_psd_wrapper(new psdonnx::PsdWrapper());
    ptr_psd_wrapper->load_model(cpl.pcr_model_path_, cpl.psd_model_path_);

    std::unique_ptr<psdonnx::JsonDataset> ptr_dataset(new psdonnx::JsonDataset(cpl.dataset_path_));
    ptr_dataset->init_reader();
    SvcPairedImages_t pis;
    while(ptr_dataset->load(pis.time, pis.img_front, pis.img_left, pis.img_right, pis.img_rear)){
        // fprintf(stderr, "load dataset t: %f, h: %d, w: %d\n", pis.time, pis.img_front.rows, pis.img_rear.cols);
        ptr_ipm_composer->Compose(pis);
        psdonnx::Detections_t det;
	    std::string psd_save_path = cpl.output_path_ + "/" + std::to_string(pis.time) + "_psd.png";
	    // ptr_psd_wrapper->run_model(pis.img_ipm, det, true, psd_save_path);
        ptr_psd_wrapper->run_model(pis.img_ipm, det, false, psd_save_path);
    }
    ptr_dataset->close_reader();
}

int main(int argc, char* argv[]){
    fprintf(stderr, "\n@i@ --> KittiFlow launched.\n");
    for(int i = 0; i < argc; i++){
        fprintf(stderr, "argv[%d] = %s\n", i, argv[i]);
    }

    is_cuda_avaliable();

    const std::string base_dir = "/mnt/c/work/github/ApaDemo/";
    const std::string dataset_dir = base_dir + "dataset";
    const std::string pcr_model_path = base_dir + "model/pcr.onnx";
    const std::string psd_model_path = base_dir + "model/psd.nms.onnx";

    // psdonnx::JsonDataset jds(dataset_dir);
    // jds.test_reader();
    do_work();

    return 0;
}
