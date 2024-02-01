#pragma once
#include <cstdio>
#include <iomanip>
#include <fstream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/profiling/profiler.h"

#include "../include/settings.h"
#include "../utils/ptime.h"
#include "../utils/singleton.h"
#include "delegates.h"

static int DEBUG = 0;

using namespace tflite;

class TfliteNetRun {
public:
    TfliteNetRun(): modify_delegate(false){};

    int model_init(const char* model_file);

    template <typename Type> 
    int model_inference();

    int model_deinit();

private:
    bool createDelegate(); 

private:
    std::unique_ptr<Interpreter> interpreter;
    DelegateProviders delegate_providers;

    std::vector<int> in_index;
    std::vector<int> out_index;
    bool modify_delegate;
};


template <typename Type>
int TfliteNetRun::model_inference() {
    Settings& s = *Settings::get();

    // fill input buffer
    std::sort(s.input_file.begin(), s.input_file.end(),
          [](const RawImagePtr& a, const RawImagePtr& b) {
              return a->getFileSize() < b->getFileSize();
          });

    for (auto index : in_index) {
        size_t num_input_elements = interpreter->tensor(index)->bytes;
    
        auto it = std::lower_bound(s.input_file.begin(), s.input_file.end(),
                                   num_input_elements,
                                   [](const RawImagePtr& a, size_t size) {
                                       return a->getFileSize() < size;
                                   });
    
        if (it == s.input_file.end() || (*it)->getFileSize() != num_input_elements) {
            std::stringstream ss_input;
            ss_input << "Error: input file is wrong, expected file size is ";
            for (auto index : in_index) {
                ss_input << interpreter->tensor(index)->bytes << ",";
            }
            LOG("{}", ss_input.str());

            std::stringstream ss_current;
            ss_current << "Error: current file size is ";
            for (const auto& inputPtr : s.input_file) {
                ss_current << inputPtr->getFileSize() << ",";
            }
            LOG("{}", ss_current.str());
            return -1;
        }
    
        memcpy(reinterpret_cast<char*>(interpreter->typed_tensor<Type>(index)), 
                                      (*it)->getAddr(), 
                                      num_input_elements);
    }
    


    // Debug
    if (DEBUG) {
        for (auto index : in_index) {
            std::ofstream file("/data/lihc/test/in.raw", std::ios::binary);
            if (file.is_open()) {
                file.write(reinterpret_cast<char*>(interpreter->typed_tensor<float>(index)), 
                                                   interpreter->tensor(index)->bytes);
            }
            file.close();
        }
    }
    
    {
        ptime p("invoke");
        // Run inference
        if(interpreter->Invoke() != kTfLiteOk)
        {
            LOG("Error: Error Invoke");
            return -1;
        }
    }

    // Read output buffers
    int n = 1;
    for (auto index : out_index) {
        size_t num_output_elements = interpreter->tensor(index)->bytes;
#if 1
        // Write the output to Settings
        std::unique_ptr<RawImage> rawImagePtr(new RawImage);
        rawImagePtr->allocBuffer(num_output_elements);
        
        Type* output = interpreter->typed_tensor<Type>(index);
        
        memcpy(rawImagePtr->getAddr(), reinterpret_cast<char*>(output), num_output_elements);
        s.output_file.push_back(std::move(rawImagePtr));
#else
        std::string output_name = s.output_path + "/" + s.outputName + std::to_string(n) + ".raw"; 
        std::ofstream outfile(output_name, std::ios::binary);

        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<char*>(interpreter->typed_tensor<Type>(index)), num_output_elements);

            outfile.close();
        } else {
            return -1;
        }
        ++n;
#endif
    }

    return 0;
}
