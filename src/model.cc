#include "model.h"

bool TfliteNetRun::createDelegate() {
    Settings& s = *Settings::get();
    auto delegates = delegate_providers.CreateAllDelegates();
    for (auto& delegate : delegates) {
        const auto delegate_name = delegate.provider->GetName();
        if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
            Logger::log("Error: Failed to apply {} delegate.", delegate_name);
            modify_delegate = false;
        } else {
            Logger::log("Applied {} delegate.", delegate_name);
            modify_delegate = true;
        }
    }

    return modify_delegate;
}

int TfliteNetRun::model_init(const char* model_file) {
    Settings& s = *Settings::get();
    delegate_providers.MergeSettingsIntoParams();
    delegate_providers.check();
    {
        ptime p("model init");
        // Load model
        static std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(model_file);
        if(model == nullptr)
        {
            Logger::log("Error: Error open model");
            return -1;
        }
        static tflite::ops::builtin::BuiltinOpResolver resolver;
        // Build the interpreter
        static InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        if(interpreter == nullptr)
        {
            Logger::log("Error: Error get interpreter");
            return -1;
        }

        if (!createDelegate()) {
            Logger::log("no delegate, use CPU");
        }

        if (!modify_delegate) {
            // Allocate tensor buffers.
            if(interpreter->AllocateTensors() != kTfLiteOk)
            {
                Logger::log("Error: no delegate, use CPU");
                return -1;
            }
        }
    }

    //interpreter->SetAllowFp16PrecisionForFp32(true);
    //interpreter->SetNumThreads(1);

    in_index = interpreter->inputs();
    out_index = interpreter->outputs();

    Logger::log("number of input is {}", interpreter->inputs().size());
    Logger::log("number of output is {}", interpreter->outputs().size());

    return 0;
}

int TfliteNetRun::model_deinit() {
    ptime p("model deinit");
    return 0;
}
