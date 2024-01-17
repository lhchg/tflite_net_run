#pragma once

using namespace std;

struct Settings {
    const char* model_name;
    const char* input_file;
    const char* output_file;
    int gpu_delegate = false;
    int nnapi_delegate = false;
    int hexagon_delegate = false;
    int xnnpack_delegate = false;
    bool allow_fp16 = true;
    bool gpu_enable_quant = false;
    bool gpu_sustained_speed = true;
    bool nnapi_burst_mode = true;
    bool nnapi_allow_dynamic = false;
    const char* nnapi_execution_priority = "high";
    bool profiling = false;
    int number_of_threads = 1;
};
