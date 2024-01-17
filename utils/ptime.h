#pragma once

#include <chrono>
#include <iostream>
#include <string.h>

#include "log.h"

class ptime
{
    // Type aliases to make accessing nested type easier
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1> >;
    using milli = std::chrono::milliseconds;
    std::chrono::time_point<clock_t> m_beg;
    std::string m_func_name;
public:
    ptime()
    {
        m_func_name = "default";
        m_beg = clock_t::now();
    }

    ptime(const std::string& func_name)
    {
        m_func_name = func_name;
        m_beg = clock_t::now();
    }
    ~ptime()
    {
        LOGD("elasp time(ms):%lld,   test func:%s\n", std::chrono::duration_cast<milli>(clock_t::now() - m_beg).count(), m_func_name.c_str());
    }

};

