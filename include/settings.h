#pragma once
using namespace std;

struct Settings {
    const char* model_name;
    const char* input_file;
    int gpu_delegate;
    int nnapi_delegate;
    int hexagon_delegate;
    int xnnpack_delegate;
};
