#pragma once

#include <iostream>
#include <memory>
#include <streambuf>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

class spdlog_streambuf : public std::streambuf
{
public:
    spdlog_streambuf(std::shared_ptr<spdlog::logger> logger, spdlog::level::level_enum lvl)
        : logger_(logger), level_(lvl) {}

protected:
    virtual int_type overflow(int_type ch) override
    {
        if (ch != traits_type::eof())
        {
            buffer_ += static_cast<char>(ch);
            if (ch == '\n')
                flush_buffer();
        }
        return ch;
    }

    virtual int sync() override
    {
        flush_buffer();
        return 0;
    }

private:
    void flush_buffer()
    {
        if (!buffer_.empty())
        {
            logger_->log(level_, buffer_);
            buffer_.clear();
        }
    }

    std::shared_ptr<spdlog::logger> logger_;
    spdlog::level::level_enum level_;
    std::string buffer_;
};

void hook_cstream()
{
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/talos.log", 1024 * 1024 * 5, 3);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("talos", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
    static spdlog_streambuf cout_buf(logger, spdlog::level::info);
    static spdlog_streambuf cerr_buf(logger, spdlog::level::err);
    std::cout.rdbuf(&cout_buf);
    std::cerr.rdbuf(&cerr_buf);
}
