#pragma once

#include <string>

namespace vicon_calibration { namespace gflags {

bool IsExtension(const std::string& input, const std::string& should_be);

bool ValidateCannotBeEmpty(const char* flagname, const std::string& value);

bool ValidateDirMustExist(const char* flagname, const std::string& value);

bool ValidateFileMustExist(const char* flagname, const std::string& value);

bool ValidateJsonFileMustExist(const char* flagname, const std::string& value);

bool ValidateJsonFileMustExistOrNONE(const char* flagname, const std::string& value);

bool ValidateMustBeJson(const char* flagname, const std::string& value);

bool ValidateBagFileMustExist(const char* flagname, const std::string& value);

}} // namespace vicon_calibration::gflags