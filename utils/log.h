#pragma once
#include <android/log.h>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

#include "../include/settings.h"

namespace fs = std::filesystem;

#define LOG_TAG "TFLiteNetRun"
#define FILE_NAME "TFLite.log"


#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

//#define LOGD(...) printf(__VA_ARGS__)
//#define LOGE(...) printf(__VA_ARGS__)

class Logger {
public:
    static void log(const std::string& message) {
        static std::ofstream logFile;
        //if (!logFile.is_open()) {
        //    logFile.open(filename, std::ios::app);
        //}

        //std::time_t currentTime = std::time(nullptr);
        //char timestamp[80];
        //std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

        //logFile << "[" << timestamp << "] " << message << std::endl;

        fs::path filePath = filename;
        try {
            if (!logFile.is_open()) {
                printf("lihc_test filename = %s\n", filename.c_str());
                logFile.open(filePath, std::ios::app);
                if (!logFile) {
                    throw std::runtime_error("Failed to open log file: " + filename);
                }
            }

            std::time_t currentTime = std::time(nullptr);
            char timestamp[80];
            std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

            logFile << "[" << timestamp << "] " << message << std::endl;
            printf("write done lihc_test filename = %s\n", filename.c_str());
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    template<typename T, typename... Args>
    static void log(const std::string& format, T value, Args... args) {
        std::ostringstream oss;
        oss << value;
        std::string newFormat = format;
        newFormat.replace(newFormat.find("{}"), 2, oss.str());
        log(newFormat, args...);
    }

    static std::string filename;
};
