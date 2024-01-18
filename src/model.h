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
    int model_inference(Type **output_vals, int& output_size);

    int model_deinit();

private:
    bool createDelegate(); 
    void PrintProfilingInfo(const profiling::ProfileEvent* e,uint32_t subgraph_index, uint32_t op_index,TfLiteRegistration registration);

private:
    std::unique_ptr<Interpreter> interpreter;
    DelegateProviders delegate_providers;

    std::vector<int> in_index;
    //const std::vector<int> out_index;
    int out_index;
    bool modify_delegate;
};


template <typename Type>
int TfliteNetRun::model_inference(Type **output_vals, int& output_size) {
    Settings& s = *Settings::get();

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
            LOGE("input file is wrong, expected file size is");
            for (auto index : in_index) {
                size_t num_input_elements = interpreter->tensor(index)->bytes;
                LOGE(" %lu ,", num_input_elements);
            }
            LOGE("\n");

            LOGE("current file size is");
            for (const auto& inputPtr : s.input_file) {
                size_t inputSize = inputPtr->getFileSize();
                LOGE(" %lu ,", inputSize);
            }
            LOGE("\n");

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
    Type* output = interpreter->typed_tensor<Type>(out_index);
    size_t num_output_elements = interpreter->tensor(out_index)->bytes;
    LOGE("num_output_elements = %zu\n", num_output_elements);
    *output_vals = (Type*)malloc(num_output_elements);
    memcpy(*output_vals, output, num_output_elements);
    output_size = num_output_elements;

    return 0;
}
