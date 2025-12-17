#pragma once

#include <iostream>
#include <memory>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <streambuf>
#include <utility>

class spdlog_streambuf : public std::streambuf {
public:
    spdlog_streambuf(std::shared_ptr<spdlog::logger> logger, const spdlog::level::level_enum lvl) :
        logger_(std::move(logger)), level_(lvl) {}

protected:
    int_type overflow(const int_type ch) override {
        if (ch != traits_type::eof()) {
            buffer_ += static_cast<char>(ch);
            if (ch == '\n')
                flush_buffer();
        }
        return ch;
    }

    int sync() override {
        flush_buffer();
        return 0;
    }

private:
    void flush_buffer() {
        if (!buffer_.empty()) {
            logger_->log(level_, buffer_);
            buffer_.clear();
        }
    }

    std::shared_ptr<spdlog::logger> logger_;
    spdlog::level::level_enum level_;
    std::string buffer_;
};

inline void hook_cstream() {
    const auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/talos.log", 1024 * 1024 * 5, 3);
    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    const auto logger = std::make_shared<spdlog::logger>("talos", sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);
    static spdlog_streambuf cout_buf(logger, spdlog::level::info);
    static spdlog_streambuf cerr_buf(logger, spdlog::level::err);
    std::cout.rdbuf(&cout_buf);
    std::cerr.rdbuf(&cerr_buf);
}
