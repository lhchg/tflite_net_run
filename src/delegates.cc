#include "../include/delegates.h"
#include "../include/settings.h"

using namespace tflite;
using namespace tflite::tools;

DelegateProviders::DelegateProviders() : delegate_list_util_(&params_){
    delegate_list_util_.AddAllDelegateParams();
    delegate_list_util_.AppendCmdlineFlags(flags_);
    
    params_.RemoveParam("help");
    delegate_list_util_.RemoveCmdlineFlag(flags_, "help");
}

void DelegateProviders::MergeSettingsIntoParams(const Settings& s) {
    if (s.gpu_delegate) {
        if (!params_.HasParam("use_gpu")) {
            std::cout << "GPU delegate execution provider isn't linked or GPU "
                         "delegate isn't supported on the platform!";
        } else {
            params_.Set<bool>("use_gpu", true);
            if (params_.HasParam("gpu_inference_for_sustained_speed")) {
                params_.Set<bool>("gpu_inference_for_sustained_speed", true);
            }
            params_.Set<bool>("gpu_precision_loss_allowed", s.allow_fp16);
        }
    }

    if (s.nnapi_delegate) {
        if (!params_.HasParam("use_nnapi")) {
            std::cout << "NNAPI delegate execution provider isn't linked or NNAPI "
                         "delegate isn't supported on the platform!";
        } else {
            params_.Set<bool>("use_nnapi", true);
            params_.Set<bool>("nnapi_allow_fp16", s.allow_fp16);
        }
    }

    //if (s.hexagon_delegate) {
    //    if (!params_.HasParam("use_hexagon")) {
    //        std::cout << "Hexagon delegate execution provider isn't linked or "
    //                   "Hexagon delegate isn't supported on the platform!";
    //    } else {
    //        params_.Set<bool>("use_hexagon", true);
    //        params_.Set<bool>("hexagon_profiling", s.profiling);
    //    }
    //}

    //if (s.xnnpack_delegate) {
    //    if (!params_.HasParam("use_xnnpack")) {
    //        std::cout << "XNNPACK delegate execution provider isn't linked or "
    //                   "XNNPACK delegate isn't supported on the platform!";
    //    } else {
    //        params_.Set<bool>("use_xnnpack", true);
    //        params_.Set<int32_t>("num_threads", s.number_of_threads);
    //    }
    //}
}

void DelegateProviders::check() {
    if (params_.HasParam("use_gpu")) {
        printf("lihc_test has use_gpu\n");
    } else {
        printf("lihc_test don't has use_gpu\n");
        
    }

    if (params_.HasParam("use_nnapi")) {
        printf("lihc_test has use_nnapi\n");
    } else {
        printf("lihc_test don't has use_nnapi\n");
        
    }

    if (params_.HasParam("use_xnnpack")) {
        printf("lihc_test has use_xnnpack\n");
    } else {
        printf("lihc_test don't has use_xnnpack\n");
        
    }

    if (params_.HasParam("use_hexagon")) {
        printf("lihc_test has use_hexagon\n");
    } else {
        printf("lihc_test don't has use_hexagon\n");
        
    }
}
