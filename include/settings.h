#pragma once

#include <vector>
#include <unordered_map>
#include <fstream>

#include "../utils/singleton.h"

class RawImage {
public:
    RawImage(): RawImage("", 0){};
    RawImage(std::string name): RawImage(name, 0){};
    RawImage(std::string name, size_t size): imageName(name), fileSize(size), addr(nullptr){};

    void allocBuffer(size_t size) {
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

    bool saveImage(std::string fileName) {
        if (addr != nullptr && fileSize != 0) { 
            std::ofstream outfile(fileName, std::ios::binary);
            if (outfile.is_open()) {
                outfile.write(addr, fileSize);
                outfile.close();
                return true;
            } else {
                return false;
            }
        }
        return false;
    }

    bool saveImage() {
        if (addr != nullptr && fileSize != 0) { 
            std::ofstream outfile(imageName, std::ios::binary);
            if (outfile.is_open()) {
                outfile.write(addr, fileSize);
                outfile.close();
                return true;
            } else {
                return false;
            }
        }
        return false;
    }

    ~RawImage(){
        if (addr != nullptr) {
            delete[] addr;
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
    ~Settings() = default;
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
    bool allow_fp16 = false;
    bool gpu_enable_quant = false;
    bool gpu_sustained_speed = false;
    bool nnapi_burst_mode = true;
    bool nnapi_allow_dynamic = false;
    std::string nnapi_execution_priority = "high";
    bool profiling = false;
    int number_of_threads = 1;
};
