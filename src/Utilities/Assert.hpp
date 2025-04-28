#pragma once

#include <cassert>
#include <iostream>

enum class AssertLevel { WARN, CRITICAL, NONE };

namespace linear_algebra
{

inline void L_ASSERT(bool condition, const char* message, AssertLevel level = AssertLevel::CRITICAL) {
    if (!condition) {
        switch (level) {
            case AssertLevel::WARN:
                std::cerr << "Warning: " << message << std::endl;
                break;
            case AssertLevel::CRITICAL:
                std::cerr << "Critical Error: " << message << std::endl;
                assert(condition);
                break;
            case AssertLevel::NONE:
                break;
        }
    }
}

}