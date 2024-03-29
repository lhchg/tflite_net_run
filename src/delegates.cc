#include "delegates.h"
#include "../utils/log.h"

using namespace tflite;
using namespace tflite::tools;

DelegateProviders::DelegateProviders() : delegate_list_util_(&params_){
    delegate_list_util_.AddAllDelegateParams();
    delegate_list_util_.AppendCmdlineFlags(flags_);
    
    params_.RemoveParam("help");
    delegate_list_util_.RemoveCmdlineFlag(flags_, "help");
}

void DelegateProviders::MergeSettingsIntoParams() {
    Settings& s = *Settings::get();
    if (s.gpu_delegate) {
        if (!params_.HasParam("use_gpu")) {
            LOG("Error: GPU delegate execution provider isn't linked or GPU delegate isn't supported on the platform");
        } else {
            params_.Set<bool>("use_gpu", true);
            if (params_.HasParam("gpu_inference_for_sustained_speed")) {
                params_.Set<bool>("gpu_inference_for_sustained_speed", s.gpu_sustained_speed);
            }
            if (params_.HasParam("gpu_experimental_enable_quant")) {
                params_.Set<bool>("gpu_experimental_enable_quant", s.gpu_enable_quant);
            }
            if (params_.HasParam("gpu_precision_loss_allowed")) {
                params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
            }
        }
    }

    if (s.nnapi_delegate) {
        if (!params_.HasParam("use_nnapi")) {
            LOG("Error: NNAPI delegate execution provider isn't linked or NNAPI delegate isn't supported on the platform!");
        } else {
            params_.Set<bool>("use_nnapi", true);
            params_.Set<std::string>("nnapi_execution_preference", "fast_single_answer");
            params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
            params_.Set<bool>("disable_nnapi_cpu", true);
            params_.Set<bool>("nnapi_use_burst_mode", s.nnapi_burst_mode);
            params_.Set<bool>("nnapi_allow_dynamic_dimensions", s.nnapi_allow_dynamic);
            params_.Set<std::string>("nnapi_execution_priority", s.nnapi_execution_priority);
        }
    }

    if (s.hexagon_delegate) {
        if (!params_.HasParam("use_hexagon")) {
            LOG("Error: Hexagon delegate execution provider isn't linked or Hexagon delegate isn't supported on the platform!");
        } else {
            params_.Set<bool>("use_hexagon", true);
            params_.Set<bool>("hexagon_profiling", s.profiling);
        }
    }

    if (s.xnnpack_delegate) {
        if (!params_.HasParam("use_xnnpack")) {
            LOG("Error: XNNPACK delegate execution provider isn't linked or XNNPACK delegate isn't supported on the platform!");
        } else {
            params_.Set<bool>("use_xnnpack", true);
            params_.Set<int32_t>("num_threads", s.number_of_threads);
        }
    }
}

void DelegateProviders::check() {
    Settings& s = *Settings::get();
    if (params_.HasParam("use_gpu")) {
        LOG("gpu delegate is supported on the platform");
    } else {
        LOG("gpu delegate is not supported on the platform");
        
    }

    if (params_.HasParam("use_nnapi")) {
        LOG("nnapi delegate is supported on the platform");
    } else {
        LOG("nnapi delegate is not supported on the platform");
        
    }

    if (params_.HasParam("use_xnnpack")) {
        LOG("xnnpack delegate is supported on the platform");
    } else {
        LOG("xnnpack delegate is not supported on the platform");
        
    }

    if (params_.HasParam("use_hexagon")) {
        LOG("hexagon delegate is supported on the platform");
    } else {
        LOG("hexagon delegate is not supported on the platform");
    }
}
