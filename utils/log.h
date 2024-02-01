#pragma once
#include "log_opt.h"

#define FILE_NAME "TFLite.log"

#define INIT_LOG_FILE(filename)\
{\
    Logger::getInstance(filename, 20);\
}

#define LOG(...)\
{\
    Logger::getInstance()->write(__VA_ARGS__);\
}
