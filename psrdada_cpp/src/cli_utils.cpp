#include "psrdada_cpp/cli_utils.hpp"

namespace psrdada_cpp {
    key_t string_to_key(std::string const& in)
    {
        key_t key;
        std::stringstream converter;
        converter << std::hex << in;
        converter >> key;
        return key;
    }

    std::string key_to_string(key_t const& in)
    {
        std::stringstream converter;
        converter << std::hex << in;
        return converter.str();
    }

    void set_log_level(std::string level)
    {
        using namespace boost::log;
        if (level == "debug")
        {
            core::get()->set_filter(trivial::severity >= trivial::debug);
        }
        else if (level == "info")
        {
            core::get()->set_filter(trivial::severity >= trivial::info);
        }
        else if (level == "warning")
        {
            core::get()->set_filter(trivial::severity >= trivial::warning);
        }
        else
        {
            core::get()->set_filter(trivial::severity >= trivial::error);
        }
    }
} //namespace
