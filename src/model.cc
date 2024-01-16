#include "../include/model.h"

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

void TfliteNetRun::PrintProfilingInfo(const profiling::ProfileEvent* e,
                        uint32_t subgraph_index, uint32_t op_index,
                        TfLiteRegistration registration) {

  std::cout << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Subgraph " << std::setw(3) << std::setprecision(3)
            << subgraph_index << ", Node " << std::setw(3)
            << std::setprecision(3) << op_index << ", OpCode " << std::setw(3)
            << std::setprecision(3) << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code)) << std::endl;
}
