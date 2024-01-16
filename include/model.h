#pragma once
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/delegate_options.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#include "settings.h"
#include "delegates.h"

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

private:
    std::unique_ptr<Interpreter> interpreter;
    DelegateProviders delegate_providers;

    int in_index;
    int out_index;
    bool modify_delegate;
};

bool TfliteNetRun::createDelegate(Settings s) {
    auto delegates = delegate_providers.CreateAllDelegates();
    for (auto& delegate : delegates) {
        const auto delegate_name = delegate.provider->GetName();
        if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
            std::cout << "Failed to apply " << delegate_name << " delegate." << std::endl;
            modify_delegate = false;
        } else {
            std::cout << "Applied " << delegate_name << " delegate." << std::endl;
            modify_delegate = true;
        }
    }

    return modify_delegate;
}

int TfliteNetRun::model_init(const char* model_file, Settings s) {
    delegate_providers.MergeSettingsIntoParams(s);
    delegate_providers.check();
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

    if (!createDelegate(s)) {
        fprintf(stderr, "Error use delegate, fall back to CPU\n");
    }

    if (!modify_delegate) {
        // Allocate tensor buffers.
        printf("lihc_test AllocateTensors\n");
        if(interpreter->AllocateTensors() != kTfLiteOk)
        {
            fprintf(stderr, "Error AllocateTensors\n");
            return -1;
        }
    }


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
