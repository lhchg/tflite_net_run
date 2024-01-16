#include <vector>
#include <fstream>
#include <iostream>

#include <getopt.h>

#include "include/model.h"
#include "include/delegates.h"
#include "include/settings.h"

int saveOutput(const char* output, int output_size) {
    std::ofstream outfile("/data/lihc/test/output", std::ios::binary);

    if (outfile.is_open()) {
        outfile.write(output, output_size);
        outfile.close();
        printf("output write success\n");
    } else {
        printf("cannot write output\n");
        return -1;
    }

    return 0;
}


char* readImg(const std::string& filename, int& file_size) {
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        char* buffer = new char[fileSize];

        file.read(buffer, fileSize);

        file_size = fileSize;

        file.close();
        printf("lihc_test fileSize= %d\n", file_size);

        return buffer;
    } else {
        std::cout << "File open failed." << std::endl;
        file_size = 0;
        return {};
    }
    return {};
}

void display_usage() {
  std::cout
      << "\n"
      << "\t--model_file, -m: model_name.tflite\n"
      << "\t--input_file, -i: image_name.raw\n"
      << "\t--gpu_delegate, -g: [1|0]\n"
      << "\t--nnapi_delegate, -n: [1|0]\n"
      << "\t--allow_fp16, -f: [true|false]\n"
      << "\t--help, -h: Print this help message\n";
}

void getInputFlag(Settings& s, int argc, char** argv) {
    int c;
    while (true) {
        static struct option long_options[] = {
            {"model_file", required_argument, nullptr, 'm'},
            {"input_file", required_argument, nullptr, 'i'},
            {"gpu_delegate", optional_argument, nullptr, 'g'},
            {"nnapi_delegate", optional_argument, nullptr, 'n'},
            {"allow_fp16", optional_argument, nullptr, 'f'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}
        };

        int option_index = 0;
        c = getopt_long(argc, argv, "m:i:g:n:h:", long_options, &option_index);

        if (c == -1) break;
        
        switch (c) {
            case 'm':
                s.model_name = optarg;
                printf("lihc_test %s\n", s.model_name);
                break;
            case 'i':
                s.input_file = optarg;
                printf("lihc_test %s\n", s.input_file);
                break;
            case 'g':
                s.gpu_delegate = strtol(optarg, nullptr, 10);
                printf("lihc_test useGPU %d\n", s.gpu_delegate);
                break;
            case 'n':
                s.nnapi_delegate = strtol(optarg, nullptr, 10);
                printf("lihc_test useNNAPI %d\n", s.nnapi_delegate);
                break;
            case 'f':
                s.allow_fp16 = strtol(optarg, nullptr, 10);
                printf("lihc_test allow fp16 %d\n", s.allow_fp16);
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

void check(tflite::tools::ToolParams& params_) {
    if (params_.HasParam("use_gpu")) {
        printf("lihc_test has use_gpu\n");
    } else {
        printf("lihc_test don't has use_gpu\n");
        
    }

    if (params_.HasParam("use_nnapi")) {
        printf("lihc_test has use_nnapi\n");
    } else {
        printf("lihc_test don't has use_nnapi\n");
        
    }

    if (params_.HasParam("use_xnnpack")) {
        printf("lihc_test has use_xnnpack\n");
    } else {
        printf("lihc_test don't has use_xnnpack\n");
        
    }

    if (params_.HasParam("use_hexagon")) {
        printf("lihc_test has use_hexagon\n");
    } else {
        printf("lihc_test don't has use_hexagon\n");
        
    }
}

void initDelegates() {
    tflite::tools::ToolParams params_;
    ProvidedDelegateList delegate_list_util_(&params_);

    delegate_list_util_.AddAllDelegateParams();

    check(params_);
    delegate_list_util_.CreateAllRankedDelegates();  
}

int main(int argc, char **argv) {
    Settings s;
    getInputFlag(s, argc, argv);
    //initDelegates();

    TfliteNetRun tfliterun;

    const char *model_file = s.model_name;
    const char *in_file = s.input_file;

    int input_size;
    int output_size;
    char* input = readImg(in_file, input_size);
    float* output;

    tfliterun.model_init(model_file, s);
    tfliterun.model_inference<float>(reinterpret_cast<float*>(input), input_size, &output, output_size);
    tfliterun.model_deinit();

    if (output != nullptr) {
        saveOutput(reinterpret_cast<char*>(output), output_size);
    } else {
        printf("output is nullptr\n");
    }

    delete [] output;
    delete [] input;

    return 0;
}
