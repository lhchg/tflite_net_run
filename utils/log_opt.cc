#include <ctime>

#include "log_opt.h"

std::unique_ptr<Logger> Logger::instance = nullptr;
std::mutex Logger::m_mutex;

std::unique_ptr<Logger>& Logger::getInstance(const std::string& filename, std::size_t bufferSize) {
    if (instance == nullptr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (instance == nullptr) {
            instance = std::make_unique<Logger>(filename, bufferSize);
        }
    }
    return instance;
}


std::unique_ptr<Logger>& Logger::getInstance() {
    if (instance == nullptr) {
        instance = std::move(getInstance("./output.log", 100));
    }
    return instance;
}

void Logger::write(const std::string message) {
    std::time_t currentTime = std::time(nullptr);
    char timestamp[80];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

    std::ostringstream mes; 
    mes << "[" << timestamp << "] " << message; 

    m_buffer.push_back(mes.str());
    if (m_buffer.size() >= m_bufferSize) {
        flush();
    }
}


Logger::~Logger() {
    flush();
    m_file.close();
}

Logger::Logger(std::string filename, std::size_t bufferSize) 
    : m_bufferSize(bufferSize)
{
    try {
        m_file.open(filename);
    
        if (!m_file.is_open()) {
            std::string errorMessage = "Failed to open file: " + filename;
            throw std::runtime_error(errorMessage);
        }
    } catch (const std::runtime_error& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
}

void Logger::flush() {
    for (const auto& message : m_buffer) {
        m_file << message << std::endl;
    }
    m_buffer.clear();
}
