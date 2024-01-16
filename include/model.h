#pragma once
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

using namespace tflite;

class TfliteNetRun {
public:
    TfliteNetRun():in_index(-1), out_index(-1){};

    int model_init(const char* model_file);

    template <typename Type> 
    int model_inference(Type *input_vals, int input_size, Type **output_vals, int& output_size);

    int model_deinit();

private:
    std::unique_ptr<Interpreter> interpreter;
    //ProvidedDelegateList delegate_list_util;
    //tflite::tools::ToolParams params;
    int in_index;
    int out_index;
};

int TfliteNetRun::model_init(const char* model_file) {
    // Load model
    static std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_file);
    if(model == nullptr)
    {
        fprintf(stderr, "Error open model\n");
        return -1;
    }
    static tflite::ops::builtin::BuiltinOpResolver resolver;
    // Build the interpreter
    static InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    if(interpreter == nullptr)
    {
        fprintf(stderr, "Error get interpreter\n");
        return -1;
    }

    //interpreter->SetAllowFp16PrecisionForFp32(true);
    //interpreter->SetNumThreads(1);
    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    options.is_precision_loss_allowed = true;
    options.serialization_dir = nullptr;
    options.model_token = nullptr;

    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

    // Allocate tensor buffers.
    //if(interpreter->AllocateTensors() != kTfLiteOk)
    //{
    //    fprintf(stderr, "Error AllocateTensors\n");
    //    return -1;
    //}

    in_index = interpreter->inputs()[0];
    out_index = interpreter->outputs()[0];
    printf("in index:%d,out index:%d\n",in_index,out_index);

    return 0;
}

int TfliteNetRun::model_deinit() {
    return 0;
}

template <typename Type>
int TfliteNetRun::model_inference(Type *input_vals, int input_size, Type **output_vals, int& output_size) {

    size_t num_input_elements = interpreter->tensor(in_index)->bytes;
    printf("num_input_elements =%zu\n", num_input_elements);
    memcpy(interpreter->typed_tensor<Type>(in_index), input_vals, num_input_elements);

    // Run inference
    if(interpreter->Invoke() != kTfLiteOk)
    {
        fprintf(stderr, "Error Invoke\n");
        return -1;
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
