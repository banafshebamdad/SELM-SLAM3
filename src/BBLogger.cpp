/*
 * Author: Banafshe Bamdad
 * Created on Mon Feb 05 2024 20:46:04 CET
 *
 */

#include "BBLogger.hpp"
#include <stdexcept> // For std::runtime_error

std::string BBLogger::s_filename;

BBLogger& BBLogger::getInstance() {
    static BBLogger instance;
    return instance;
}

void BBLogger::setFilename(const std::string& filename) {
    s_filename = filename;
}

void BBLogger::log(const std::string& message) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (!m_logFile.is_open()) {
        m_logFile.open(s_filename, std::ios::app);
        if (!m_logFile.is_open()) {
            throw std::runtime_error("BB Unable to open BBLogger log file: " + s_filename);
        }
    }
    m_logFile << message << std::endl;
}

BBLogger::BBLogger() {
}

BBLogger::~BBLogger() {
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
}
