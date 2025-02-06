/*
 * Author: Banafshe Bamdad
 * Created on Mon Feb 05 2024 21:08:24 CET
 *
 */

#ifndef BBLOGGER_H
#define BBLOGGER_H

#include <fstream>
#include <mutex>
#include <string>

class BBLogger {
public:
    static BBLogger& getInstance();
    static void setFilename(const std::string& filename);
    void log(const std::string& message);

private:
    static std::string s_filename;
    std::ofstream m_logFile;
    std::mutex m_mutex;
    BBLogger(); 
    ~BBLogger();
    BBLogger(const BBLogger&) = delete;
    BBLogger& operator=(const BBLogger&) = delete;
};

#endif // BBLOGGER_H
