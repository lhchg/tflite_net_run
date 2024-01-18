#include <vector>
#include <fstream>
#include <iostream>

#include <getopt.h>

#include "src/model.h"
#include "src/delegates.h"
#include "include/settings.h"
#include "utils/log.h"

int saveOutput() {
    Settings& s = *Settings::get();

    int n = 1;
    for (const auto& output : s.output_file) {
        std::string output_name = s.output_path + "/" + s.outputName + std::to_string(n) + ".raw"; 
        std::ofstream outfile(output_name, std::ios::binary);
        
        if (outfile.is_open()) {
            outfile.write(output->getAddr(), output->getFileSize());
            outfile.close();
            LOGD("output write success\n");
        } else {
            LOGD("cannot write output\n");
            return -1;
        }
        ++n;
    }

    return 0;
}


char* readImg(const std::string& filename, size_t& file_size) {
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        char* buffer = new char[fileSize];

        file.read(buffer, fileSize);

        file_size = fileSize;

        file.close();
        LOGD("fileSize= %lu\n", file_size);

        return buffer;
    } else {
        LOGE("input file (%s) open failed.\n", filename.c_str());
        file_size = 0;
        return nullptr;
    }
    return nullptr;
}

uint64_t getFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);


        file.close();
        LOGD("%s fileSize= %lu\n", filename.c_str(), (size_t)fileSize);

        return fileSize;
    } else {
        LOGE("input file (%s) open failed.\n", filename.c_str());
        return 0;
    }

    return 0;
}

void fillImage(const std::string fileName) {
    Settings& s = *Settings::get();

    size_t size = 0;
    char* buffer = readImg(fileName, size);

    std::unique_ptr<RawImage> rawImagePtr(new RawImage());    
    rawImagePtr->setImageName(fileName);
    rawImagePtr->allocBuffer(size);
    if (buffer != nullptr) {
        memcpy(rawImagePtr->getAddr(), buffer, size * sizeof(char));
    } else {
        exit(-1);
    }
    s.input_file.push_back(std::move(rawImagePtr));

    delete[] buffer;
}

void parseFileNames(const std::string& fileNames) {

    std::istringstream iss(fileNames);
    std::string fileName;

    while (std::getline(iss, fileName, ',')) {
        fillImage(fileName);
    }
}

void display_usage() {
  std::cout
      << "\t--model_file, -m: file path of model\n"
      << "\t--input_file, -i: file path of input file, \n"
      << "\t                  if more than one file, use ',' to separate\n"
      << "\t                  e.g. input1.raw,input2.raw\n"
      << "\t--output_file, -o: file path of output file\n"
      << "\t--gpu_delegate, -g: use gpu delegate or not [1|0]\n"
      << "\t--nnapi_delegate, -n: use nnapi delegate or not [1|0]\n"
      << "\t--allow_fp16, -f: Whether to allow the GPU/NNAPI delegate to carry out computation \n"
      << "\t                  with some precision loss (i.e. processing in FP16) or not. If allowed, \n"
      << "\t                  the performance will increase (default=true) [1|0]\n"
      << "\t--gpu_enable_quant, -q: Whether to allow the GPU delegate to run a 8-bit quantized \n"
      << "\t                        model or not (default=false) [1|0]\n"
      << "\t--gpu_sustained_speed, s: Whether to prefer maximizing the throughput. This mode will help when the\n"
      << "\t                          same delegate will be used repeatedly on multiple inputs. This is supported\n"
      << "\t                          on non-iOS platforms. (default=true) [1|0]\n"
      << "\t--help, -h: Print this help message\n";
}

void getInputFlag(int argc, char** argv) {
    Settings& s = *Settings::get();
    const char* inputFiles = nullptr;
    int c;
    while (true) {
        static struct option long_options[] = {
            {"model_file", required_argument, nullptr, 'm'},
            {"input_file", required_argument, nullptr, 'i'},
            {"output_path", optional_argument, nullptr, 'o'},
            {"gpu_delegate", optional_argument, nullptr, 'g'},
            {"nnapi_delegate", optional_argument, nullptr, 'n'},
            {"allow_fp16", optional_argument, nullptr, 'f'},
            {"gpu_enable_quant", optional_argument, nullptr, 'q'},
            {"gpu_sustained_speed", optional_argument, nullptr, 's'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}
        };

        int option_index = 0;
        c = getopt_long(argc, argv, "m:i:o:g:n:f:q:s:h:", long_options, &option_index);

        if (c == -1) break;
        
        switch (c) {
            case 'm':
                s.model_name = optarg;
                LOGD("model name is %s\n", s.model_name.c_str());
                break;
            case 'i':
                inputFiles = optarg;
                parseFileNames(inputFiles);
                LOGD("input file number is %lu\n", s.input_file.size());
                break;
            case 'o':
                s.output_path = optarg;
                LOGD("output file is %s\n", s.output_path.c_str());
                break;
            case 'g':
                s.gpu_delegate = strtol(optarg, nullptr, 10);
                LOGD("use gpu delegate %d\n", s.gpu_delegate);
                break;
            case 'n':
                s.nnapi_delegate = strtol(optarg, nullptr, 10);
                LOGD("use NNAPI delegate %d\n", s.nnapi_delegate);
                break;
            case 'f':
                s.allow_fp16 = strtol(optarg, nullptr, 10);
                LOGD("allow fp16 %d\n", s.allow_fp16);
                break;
            case 'q':
                s.gpu_enable_quant = strtol(optarg, nullptr, 10);
                LOGD("gpu_enable_quant %d\n", s.gpu_enable_quant);
                break;
            case 's':
                s.gpu_sustained_speed = strtol(optarg, nullptr, 10);
                LOGD("gpu_sustained_speed %d\n", s.gpu_sustained_speed);
                break;
            case 'h':
            case '?':
                display_usage();
                exit(-1);
            default:
                exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    Settings& s = *Settings::get();
    getInputFlag(argc, argv);

    TfliteNetRun tfliterun;

    std::string model_file = s.model_name;

    tfliterun.model_init(model_file.c_str());
    tfliterun.model_inference<float>();
    tfliterun.model_deinit();

    saveOutput();

    Settings::release();

    return 0;
}
