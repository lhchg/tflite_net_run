#pragma once

#include <vector>
#include <unordered_map>

#include "../utils/singleton.h"

class RawImage {
public:
    RawImage(): imageName(""), addr(nullptr), fileSize(0){};

    void allocBuffer(size_t size) {
        printf("lihc_test allocBuffer size is %lu\n", size);
        addr = new char[size];
        fileSize = size;
    }

    const std::string getImageName() {
        return imageName;
    }

    void setImageName(const std::string name) {
        imageName = name;
    }

    size_t getFileSize() {
        return fileSize;
    }

    char* getAddr() {
        return addr;
    }

    ~RawImage(){
        printf("lihc_test ~RawImage\n");
        if (addr != nullptr) {
            delete[] addr;
        } else {
            printf("lihc_test addr is nullptr\n");
        }

        addr = nullptr;
    };

private:
    std::string imageName;
    char* addr;
    size_t fileSize;
};

using RawImagePtr = std::unique_ptr<RawImage>;
using RawImageList = std::vector<RawImagePtr>; 

class Settings : public Singleton<Settings> {
public:
    Settings() = default;
    ~Settings() {
        printf("lihc_test ~Settings\n");
    }
public:
    std::string model_name;

    RawImageList input_file;
    RawImageList output_file;

    std::string output_path = "output";
    std::string outputName = "output";


    int gpu_delegate = false;
    int nnapi_delegate = false;
    int hexagon_delegate = false;
    int xnnpack_delegate = false;
    bool allow_fp16 = true;
    bool gpu_enable_quant = false;
    bool gpu_sustained_speed = true;
    bool nnapi_burst_mode = true;
    bool nnapi_allow_dynamic = false;
    std::string nnapi_execution_priority = "high";
    bool profiling = false;
    int number_of_threads = 1;
};
