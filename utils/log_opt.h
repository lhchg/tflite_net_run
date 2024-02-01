#pragma once
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <mutex>


class Logger{
public:
    static std::unique_ptr<Logger>& getInstance(const std::string& filename, std::size_t bufferSize);

    static std::unique_ptr<Logger>& getInstance();

    ~Logger();

    Logger(std::string filename, std::size_t bufferSize);

    void write(const std::string message);

    template<typename T, typename... Args>
    void write(const std::string message, T value, Args... args) {
        std::ostringstream oss;
        oss << value;
        std::string newString = message;
        newString.replace(newString.find("{}"), 2, oss.str());
        write(newString, args...);
    }

private:
    void flush();

private:
    std::vector<std::string> m_buffer;
    std::ofstream m_file;
    std::size_t m_bufferSize;
    static std::unique_ptr<Logger> instance;
    static std::mutex m_mutex;
};



