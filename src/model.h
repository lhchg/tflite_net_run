#pragma once
#include <cstdio>
#include <iomanip>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/profiling/profiler.h"

#include "../include/settings.h"
#include "delegates.h"
#include "../utils/ptime.h"

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
    void PrintProfilingInfo(const profiling::ProfileEvent* e,uint32_t subgraph_index, uint32_t op_index,TfLiteRegistration registration);

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
    for (auto index : in_index) {
        size_t num_input_elements = interpreter->tensor(index)->bytes;

        auto input = s.input_file.begin();
        for (; input != s.input_file.end(); input++) {
            if (input->get()->getFileSize() == num_input_elements) {
                memcpy(interpreter->typed_tensor<Type>(index), input->get()->getAddr(), num_input_elements);
                continue;
            }
        }
        if (input == s.input_file.end()) {
            std::stringstream ss_input;
            ss_input << "input file is wrong, expected file size is";
            for (auto index : in_index) {
                size_t num_input_elements = interpreter->tensor(index)->bytes;
                ss_input << num_input_elements << ",";
            }
            LOGE("%s\n", ss_input.str().c_str());

            std::stringstream ss_current;
            ss_current << "current file size is";
            for (const auto& inputPtr : s.input_file) {
                size_t inputSize = inputPtr->getFileSize();
                ss_current << inputSize << ",";
            }
            LOGE("%s\n", ss_current.str().c_str());

            return -1;
        }
    }

    {
        ptime p("invoke");
        // Run inference
        if(interpreter->Invoke() != kTfLiteOk)
        {
            LOGE("Error Invoke\n");
            return -1;
        }
    }

    // Read output buffers
    for (auto index : out_index) {
        size_t num_output_elements = interpreter->tensor(index)->bytes;
        std::unique_ptr<RawImage> rawImagePtr(new RawImage);
        rawImagePtr->allocBuffer(num_output_elements);
        
        Type* output = interpreter->typed_tensor<Type>(index);
        
        memcpy(rawImagePtr->getAddr(), output, num_output_elements);
        s.output_file.push_back(std::move(rawImagePtr));
    }

    return 0;
}
