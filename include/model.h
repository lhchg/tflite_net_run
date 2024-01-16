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

#include "settings.h"
#include "delegates.h"
#include "ptime.h"

using namespace tflite;

class TfliteNetRun {
public:
    TfliteNetRun():in_index(-1), out_index(-1), modify_delegate(false){};

    int model_init(const char* model_file, Settings s);

    template <typename Type> 
    int model_inference(Type *input_vals, int input_size, Type **output_vals, int& output_size);

    int model_deinit();

private:
    bool createDelegate(Settings s); 
    void PrintProfilingInfo(const profiling::ProfileEvent* e,uint32_t subgraph_index, uint32_t op_index,TfLiteRegistration registration);

private:
    std::unique_ptr<Interpreter> interpreter;
    DelegateProviders delegate_providers;

    int in_index;
    int out_index;
    bool modify_delegate;
};


template <typename Type>
int TfliteNetRun::model_inference(Type *input_vals, int input_size, Type **output_vals, int& output_size) {

    size_t num_input_elements = interpreter->tensor(in_index)->bytes;
    printf("num_input_elements =%zu\n", num_input_elements);
    memcpy(interpreter->typed_tensor<Type>(in_index), input_vals, num_input_elements);

    {
        ptime p("invoke");
        // Run inference
        if(interpreter->Invoke() != kTfLiteOk)
        {
            fprintf(stderr, "Error Invoke\n");
            return -1;
        }
    }


    // Read output buffers
    Type* output = interpreter->typed_tensor<Type>(out_index);
    size_t num_output_elements = interpreter->tensor(out_index)->bytes;
    printf("num_output_elements = %zu\n", num_output_elements);
    *output_vals = (Type*)malloc(num_output_elements);
    memcpy(*output_vals, output, num_output_elements);
    output_size = num_output_elements;
    if (*output_vals == nullptr) {
        printf("output_vals is nullptr \n");
    } else {
        printf("output_vals is not nullptr \n");
    }

    return 0;
}
