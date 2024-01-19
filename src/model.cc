#include "model.h"

bool TfliteNetRun::createDelegate() {
    Settings& s = *Settings::get();
    auto delegates = delegate_providers.CreateAllDelegates();
    for (auto& delegate : delegates) {
        const auto delegate_name = delegate.provider->GetName();
        if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
            LOGE("Failed to apply %s delegate.\n", delegate_name.c_str());
            modify_delegate = false;
        } else {
            LOGD("Applied %s delegate.\n", delegate_name.c_str());
            modify_delegate = true;
        }
    }

    return modify_delegate;
}

int TfliteNetRun::model_init(const char* model_file) {
    delegate_providers.MergeSettingsIntoParams();
    delegate_providers.check();
    {
        ptime p("model init");
        // Load model
        static std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_file);
        if(model == nullptr)
        {
            LOGE("Error open model\n");
            return -1;
        }
        static tflite::ops::builtin::BuiltinOpResolver resolver;
        // Build the interpreter
        static InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        if(interpreter == nullptr)
        {
            LOGE("Error get interpreter\n");
            return -1;
        }

        if (!createDelegate()) {
            LOGD("no delegate, use CPU\n");
        }

        if (!modify_delegate) {
            // Allocate tensor buffers.
            if(interpreter->AllocateTensors() != kTfLiteOk)
            {
                LOGE("Error AllocateTensors\n");
                return -1;
            }
        }
    }

    //interpreter->SetAllowFp16PrecisionForFp32(true);
    //interpreter->SetNumThreads(1);

    in_index = interpreter->inputs();
    out_index = interpreter->outputs();

    LOGD("number of input is %lu\n", interpreter->inputs().size());
    LOGD("number of output is %lu\n", interpreter->outputs().size());

    return 0;
}

int TfliteNetRun::model_deinit() {
    ptime p("model deinit");
    return 0;
}
