#pragma once

#include "scheduler.hpp"

#include <vector>


namespace scheduler {
    class BufferPool {
    public:
        explicit BufferPool(const size_t count, const int rows, const int cols, const int type) :
            rows_(rows), cols_(cols), type_(type) {
            for (size_t i = 0; i < count; ++i) {
                pool_.push_back(ImageBuffer::create(rows, cols, type));
            }
            available_.resize(count, true);
        }

        // 禁止拷贝
        BufferPool(const BufferPool &) = delete;
        BufferPool &operator=(const BufferPool &) = delete;
        // 禁止移动
        BufferPool(BufferPool &&other) noexcept = delete;
        BufferPool &operator=(BufferPool &&other) noexcept = delete;

        ImagePtr acquire() {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < available_.size(); ++i) {
                if (available_[i]) {
                    available_[i] = false;
                    return pool_[i];
                }
            }
            // Pool exhausted, create new (will be GC'd later)
            return ImageBuffer::create(rows_, cols_, type_);
        }

        void release(const ImagePtr &ptr) {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < pool_.size(); ++i) {
                if (pool_[i].get() == ptr.get()) {
                    available_[i] = true;
                    return;
                }
            }
            // Not from pool, let it go
        }

    private:
        std::vector<ImagePtr> pool_;
        std::vector<bool> available_;
        std::mutex mutex_;
        int rows_, cols_, type_;
    };
} // namespace scheduler
