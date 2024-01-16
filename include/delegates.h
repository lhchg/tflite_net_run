#pragma once
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"
#include "settings.h"

using namespace tflite::tools;
using namespace tflite;

class DelegateProviders {
public:
    DelegateProviders();

    bool InitFromCmdlineArgs(int* argc, const char** argv) {
        return Flags::Parse(argc, argv, flags_);
    }

    void MergeSettingsIntoParams(const Settings& s);

    std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates() const {
        return delegate_list_util_.CreateAllRankedDelegates();
    }

    std::string GetHelpMessage(const std::string& cmdline) const {
        return Flags::Usage(cmdline, flags_);
    }

    void check();

private:
    tflite::tools::ToolParams params_;
    
    ProvidedDelegateList delegate_list_util_;
    
    std::vector<tflite::Flag> flags_;
};
