#ifndef VIWO_UTILS_H
#define VIWO_UTILS_H

#include <chrono>
#include <memory>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#define HANG_STOPWATCH() auto _ViwoUtilsPtr_ = ViwoUtils::HangStopWatch(__FUNCTION__);

class ViwoUtils
{
public:
    ViwoUtils() {}
    ~ViwoUtils() {}

    static inline __attribute__((always_inline)) uint64_t CurrentMicros() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::time_point_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()).time_since_epoch()).count();
    }

    /*print function enter/exit/time_interval*/
    static std::shared_ptr<uint64_t> HangStopWatch(const char* func_name){
        uint64_t* pts_us = new uint64_t;
        *pts_us = CurrentMicros();
        fprintf(stderr, "stop watch trigger by %s (epoch %ld us)\n", func_name, *pts_us);
        return std::shared_ptr<uint64_t>(pts_us, [func_name](uint64_t* ptr){
            uint64_t ts_us = CurrentMicros();
            // LOG(INFO) << "stop watch end: " << ts_us << " us.";
            fprintf(stderr, "stop watch end by %s (elapse = %ld us)\n", func_name, (ts_us - *ptr));
            delete ptr;
        });
    }

    /* between 0~k */
    static double RandDouble(double k = 1.0f){
        return k * rand() / double(RAND_MAX);
    }

    /* unit: m/s */
    static double WheelVelBias(double k = 0.01f){
        /* this noise should keep positive or negative */
        return fabs(RandDouble(k));
    }

    /* small mat debug print */
    static std::string CvMat2Str(cv::Mat& mat){
        cv::Mat oneRow = mat.reshape(0, 1);
        std::ostringstream os;
        os << oneRow;
        return os.str();
    }

};

#endif