#pragma once

namespace vicon_calibration {

#ifndef FILENAME
#define FILENAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(M, ...) \
  fprintf(stderr, "[ERROR] [%s:%d] " M "\n", FILENAME, __LINE__, ##__VA_ARGS__)
#endif

#ifndef LOG_INFO
#define LOG_INFO(M, ...) fprintf(stdout, "[INFO] " M "\n", ##__VA_ARGS__)
#endif

#ifndef LOG_WARN
#define LOG_WARN(M, ...) fprintf(stdout, "[WARNING] " M "\n", ##__VA_ARGS__)
#endif

}  // namespace vicon_calibration
