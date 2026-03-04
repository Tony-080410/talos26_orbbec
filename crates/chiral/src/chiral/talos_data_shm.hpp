#pragma once

#include <atomic>
#include <concepts>
#include <cstring>
#include <expected>
#include <fcntl.h>
#include <filesystem>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "chrial.hpp"

namespace talos::chiral::ipc {

// ============ 编译期类型布局 Hash ============

namespace detail {

/**
 * @brief FNV-1a 64位 hash 算法（编译期友好）
 */
struct Fnv1a64 {
    static constexpr uint64_t prime        = 1099511628211ULL;
    static constexpr uint64_t offset_basis = 14695981039346656037ULL;

    constexpr static uint64_t hash(std::string_view data) noexcept {
        uint64_t h = offset_basis;
        for (char c : data) {
            h ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
            h *= prime;
        }
        return h;
    }

    constexpr static uint64_t hash_combine(uint64_t h, uint64_t value) noexcept {
        h ^= value;
        h *= prime;
        return h;
    }
};
using namespace talos::chrial;
/**
 * @brief 编译期类型布局 Hash
 *
 * 使用类型名、大小、对齐计算唯一的 hash 值。
 * 用于检测 ABI 不兼容（结构体定义变化）。
 *
 * 注意：只能检测大小和对齐变化，无法检测成员重排。
 * 如果需要更强的检测，应使用 offsetof 遍历所有成员。
 */
template <typename T>
struct TypeLayoutHash {
    static constexpr uint64_t compute() noexcept {
        uint64_t h = Fnv1a64::hash("talos-chrial");
        h          = Fnv1a64::hash_combine(h, sizeof(T));
        h          = Fnv1a64::hash_combine(h, alignof(T));
        // TODO: 加入成员偏移量（需要在编译期遍历所有成员）
        return h;
    }

    static constexpr uint64_t value = compute();
};

// 编译期断言：确保 hash 在编译期计算
static_assert(TypeLayoutHash<chrial::TalosData>::value != 0, "TypeLayoutHash must be comptime");

} // namespace detail

// ============ 强类型包装 ============

/**
 * @brief 共享内存名称（强类型防止字符串误用）
 */
class SharedMemoryName {
public:
    constexpr explicit SharedMemoryName(std::string_view name) noexcept
        : name_(name) {}

    [[nodiscard]] constexpr std::string_view view() const noexcept { return name_; }

    SharedMemoryName() = delete;

private:
    std::string_view name_;
};

/**
 * @brief 共享内存大小（强类型防止 size_t 误用）
 */
class SharedMemorySize {
public:
    constexpr explicit SharedMemorySize(size_t size) noexcept
        : size_(size) {}

    [[nodiscard]] constexpr size_t value() const noexcept { return size_; }

    [[nodiscard]] constexpr bool is_valid() const noexcept { return size_ > 0; }

    constexpr explicit operator size_t() const noexcept { return size_; }

private:
    size_t size_;
};

// ============ 错误类型 ============

enum class ShmError : uint8_t {
    OpenFailed       = 0,
    TruncateFailed   = 1,
    MapFailed        = 2,
    NotFound         = 3,
    InvalidMagic     = 4,
    VersionMismatch  = 5,
    TypeHashMismatch = 6, // 新增：类型布局 hash 不匹配
};

// ============ 常量定义 ============

inline constexpr SharedMemoryName TALOS_DATA_SHM_NAME("talos_data");
inline constexpr uint32_t TALOS_SHM_MAGIC   = 0x544C4454; // "TLDT"
inline constexpr uint32_t TALOS_SHM_VERSION = 1;

// ============ 共享内存头部 ============

/**
 * @brief 共享内存头部（原子操作保证多进程可见性）
 *
 * 布局优化：
 * - write_seq 和 read_seq 位于不同缓存行（64字节），避免 false sharing
 * - Writer 只修改 write_seq，Reader 只修改 read_seq
 */
struct alignas(64) ShmHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t type_hash; // TalosData 类型布局 hash

    // Writer 专用缓存行（偏移 16-63）
    std::atomic<uint64_t> write_seq{0}; // 已完成写入序号（Writer 递增）
    std::atomic<uint64_t> data_seq{0};  // 数据序列锁：偶数稳定，奇数写入中
    uint8_t _pad_write[32];             // 填充到 64 字节边界（16+8+8+32=64）

    // Reader 专用缓存行（偏移 64-127，第二缓存行）
    alignas(64) std::atomic<uint64_t> read_seq{0}; // 读取序号（Reader 更新）
    uint8_t _pad_read[56];                         // 填充到 64 字节边界（8+56=64）
};
static_assert(sizeof(ShmHeader) == 128);

// ============ 共享内存布局 ============

/**
 * @brief TalosData 共享内存完整布局
 */
struct TalosShm {
    ShmHeader header;
    chrial::TalosData data;
};
// 注意: 由于对齐要求，实际大小可能大于 header + data

// TalosData 编译期类型布局 hash
inline constexpr uint64_t TALOS_DATA_TYPE_HASH = detail::TypeLayoutHash<chrial::TalosData>::value;
inline constexpr SharedMemorySize TALOS_SHM_SIZE(sizeof(TalosShm));

// ============ 文件路径工具 ============

namespace detail {

[[nodiscard]] inline std::filesystem::path shm_to_path(std::string_view name) {
    if (!name.empty() && name[0] == '/') {
        name = name.substr(1);
    }
    return std::filesystem::path("/tmp") / name;
}

} // namespace detail

// ============ RAII 共享内存区域 ============

class ShmRegion {
public:
    ~ShmRegion() noexcept {
        // 析构函数绝不能抛异常！RAII 是信仰！
        try {
            if (owner_ && !path_.empty()) {
                std::error_code ec;
                std::filesystem::remove(path_, ec); // noexcept 版本
                (void)ec;                           // 忽略错误，析构时无法处理
            }
        } catch (...) {
            // 捕获所有异常，确保不会逃逸
        }

        if (data_ != nullptr && data_ != MAP_FAILED) {
            munmap(data_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    ShmRegion(ShmRegion&& other) noexcept
        : data_(std::exchange(other.data_, nullptr))
        , size_(std::exchange(other.size_, 0))
        , fd_(std::exchange(other.fd_, -1))
        , owner_(std::exchange(other.owner_, false))
        , path_(std::move(other.path_)) {}

    ShmRegion& operator=(ShmRegion&& other) noexcept {
        if (this != &other) {
            cleanup();
            data_  = std::exchange(other.data_, nullptr);
            size_  = std::exchange(other.size_, 0);
            fd_    = std::exchange(other.fd_, -1);
            owner_ = std::exchange(other.owner_, false);
            path_  = std::move(other.path_);
        }
        return *this;
    }

    ShmRegion(const ShmRegion&)            = delete;
    ShmRegion& operator=(const ShmRegion&) = delete;

    [[nodiscard]] static std::expected<ShmRegion, ShmError>
        create(SharedMemoryName name, SharedMemorySize size) {
        const auto path = detail::shm_to_path(name.view());

        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
        if (fd < 0) {
            return std::unexpected(ShmError::OpenFailed);
        }

        if (ftruncate(fd, static_cast<off_t>(size.value())) < 0) {
            close(fd);
            std::filesystem::remove(path);
            return std::unexpected(ShmError::TruncateFailed);
        }

        void* ptr = mmap(nullptr, size.value(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            std::filesystem::remove(path);
            return std::unexpected(ShmError::MapFailed);
        }

        std::memset(ptr, 0, size.value());

        return ShmRegion(ptr, size.value(), fd, true, std::move(path));
    }

    [[nodiscard]] static std::expected<ShmRegion, ShmError>
        open(SharedMemoryName name, SharedMemorySize size) {
        const auto path = detail::shm_to_path(name.view());

        const int fd = ::open(path.c_str(), O_RDWR, 0);
        if (fd < 0) {
            return std::unexpected(ShmError::NotFound);
        }

        struct stat st{};
        if (fstat(fd, &st) < 0 || static_cast<size_t>(st.st_size) < size.value()) {
            close(fd);
            return std::unexpected(ShmError::MapFailed);
        }

        void* ptr = mmap(nullptr, size.value(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            return std::unexpected(ShmError::MapFailed);
        }

        return ShmRegion(ptr, size.value(), fd, false, std::move(path));
    }

    template <typename T>
    [[nodiscard]] T* as() noexcept {
        return static_cast<T*>(data_);
    }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept {
        return static_cast<const T*>(data_);
    }

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] bool is_owner() const noexcept { return owner_; }

private:
    ShmRegion(void* data, size_t size, int fd, bool owner, std::filesystem::path path) noexcept
        : data_(data)
        , size_(size)
        , fd_(fd)
        , owner_(owner)
        , path_(std::move(path)) {}

    void cleanup() noexcept {
        if (data_ != nullptr && data_ != MAP_FAILED) {
            munmap(data_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        if (owner_ && !path_.empty()) {
            std::filesystem::remove(path_);
        }
    }

    void* data_{nullptr};
    size_t size_{0};
    int fd_{-1};
    bool owner_{false};
    std::filesystem::path path_;
};

// ============ Writer ============

/**
 * @brief TalosData 共享内存 Writer
 *
 * 零拷贝写入，使用原子序号保证新数据可见性。
 * 创建时自动写入 TalosData 类型布局 hash，防止 ABI 不兼容。
 *
 * 使用示例:
 * ```cpp
 * auto writer = TalosDataWriter::create();
 * if (!writer) {
 *     SPDLOG_ERROR("Failed to create writer: {}", magic_enum::enum_name(writer.error()));
 *     return;
 * }
 *
 * TalosData data;
 * data.gimbal_link = ...;
 * writer->write(data);
 * ```
 */
class TalosDataWriter {
public:
    ~TalosDataWriter() = default;

    TalosDataWriter(TalosDataWriter&&)                 = default;
    TalosDataWriter& operator=(TalosDataWriter&&)      = default;
    TalosDataWriter(const TalosDataWriter&)            = delete;
    TalosDataWriter& operator=(const TalosDataWriter&) = delete;

    /**
     * @brief 创建共享内存 Writer
     *
     * 初始化共享内存头部：
     * - 魔数：验证文件类型
     * - 版本号：协议版本
     * - 类型 hash：TalosData 编译期布局 hash，防止 ABI 不兼容
     */
    [[nodiscard]] static std::expected<TalosDataWriter, ShmError> create() {
        auto region = ShmRegion::create(TALOS_DATA_SHM_NAME, TALOS_SHM_SIZE);
        if (!region) {
            return std::unexpected(region.error());
        }

        auto* shm = region->as<TalosShm>();

        // 初始化头部（包含类型 hash）
        shm->header.magic     = TALOS_SHM_MAGIC;
        shm->header.version   = TALOS_SHM_VERSION;
        shm->header.type_hash = TALOS_DATA_TYPE_HASH; // 写入编译期类型 hash
        shm->header.write_seq.store(0, std::memory_order_release);
        shm->header.data_seq.store(0, std::memory_order_release);
        shm->header.read_seq.store(0, std::memory_order_release);

        return TalosDataWriter(std::move(*region));
    }

    /**
     * @brief 写入 TalosData（零拷贝）
     *
     * # Safety
     *
     * - 通过 data_seq 序列锁避免 Reader 读到撕裂数据
     * - write_seq 仅在完整写入后递增
     */
    void write(const chrial::TalosData& data) {
        auto* shm = region_.as<TalosShm>();

        // 进入写入区（奇数表示写入中）
        shm->header.data_seq.fetch_add(1, std::memory_order_seq_cst);

        std::memcpy(&shm->data, &data, sizeof(chrial::TalosData));

        // 退出写入区（偶数表示数据稳定）
        shm->header.data_seq.fetch_add(1, std::memory_order_seq_cst);

        // 完整写入后再发布新序号
        shm->header.write_seq.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] uint64_t write_seq() const noexcept {
        // 只读计数器，不需要同步其他数据，relaxed 足够
        return region_.as<TalosShm>()->header.write_seq.load(std::memory_order_relaxed);
    }

private:
    explicit TalosDataWriter(ShmRegion&& region) noexcept
        : region_(std::move(region)) {}

    ShmRegion region_;
};

// ============ Reader ============

/**
 * @brief TalosData 共享内存 Reader
 *
 * 零拷贝读取，通过原子序号检测新数据。
 * 打开时强制校验类型 hash，确保 ABI 兼容性。
 *
 * 使用示例:
 * ```cpp
 * auto reader = TalosDataReader::open();
 * if (!reader) {
 *     if (reader.error() == ShmError::TypeHashMismatch) {
 *         SPDLOG_ERROR("ABI mismatch: TalosData definition changed!");
 *     }
 *     return;
 * }
 *
 * if (const chrial::TalosData* data = reader->read_new()) {
 *     // 使用 data（零拷贝）
 * }
 * ```
 */
class TalosDataReader {
public:
    ~TalosDataReader() = default;

    TalosDataReader(TalosDataReader&&)                 = default;
    TalosDataReader& operator=(TalosDataReader&&)      = default;
    TalosDataReader(const TalosDataReader&)            = delete;
    TalosDataReader& operator=(const TalosDataReader&) = delete;

    /**
     * @brief 打开已存在的共享内存 Reader
     *
     * 强制校验：
     * 1. 魔数：验证文件类型
     * 2. 版本号：协议版本兼容性
     * 3. **类型 hash**：TalosData 布局 hash，防止 ABI 不兼容
     *
     * 如果类型 hash 不匹配，说明 Writer 和 Reader 的 TalosData 定义不一致，
     * 会导致内存布局错误，必须拒绝连接。
     */
    [[nodiscard]] static std::expected<TalosDataReader, ShmError> open() {
        auto region = ShmRegion::open(TALOS_DATA_SHM_NAME, TALOS_SHM_SIZE);
        if (!region) {
            return std::unexpected(region.error());
        }

        auto* shm = region->as<TalosShm>();

        // 验证魔数
        if (shm->header.magic != TALOS_SHM_MAGIC) {
            return std::unexpected(ShmError::InvalidMagic);
        }

        // 验证版本号
        if (shm->header.version != TALOS_SHM_VERSION) {
            return std::unexpected(ShmError::VersionMismatch);
        }

        // 强制验证类型 hash（防止 ABI 不兼容）
        if (shm->header.type_hash != TALOS_DATA_TYPE_HASH) {
            return std::unexpected(ShmError::TypeHashMismatch);
        }

        return TalosDataReader(std::move(*region));
    }

    /**
     * @brief 读取新数据（稳定快照）
     *
     * @return 如果有新数据，返回指向 Reader 内部快照的指针；否则返回 nullptr
     *
     * # Safety
     *
     * - 返回的指针指向 Reader 内部快照，下次调用 read_new() 后失效
     * - 如需持久化，请立即复制数据
     */
    [[nodiscard]] const chrial::TalosData* read_new() noexcept {
        auto* shm = region_.as<TalosShm>();

        const uint64_t write_seq = shm->header.write_seq.load(std::memory_order_acquire);

        if (write_seq == last_seq_) {
            return nullptr;
        }

        copy_consistent_snapshot(*shm);
        last_seq_ = write_seq;
        shm->header.read_seq.store(write_seq, std::memory_order_release);

        return &snapshot_;
    }

    /**
     * @brief 读取最新数据（即使已读取过）
     */
    [[nodiscard]] const chrial::TalosData* read_latest() noexcept {
        auto* shm = region_.as<TalosShm>();
        copy_consistent_snapshot(*shm);
        return &snapshot_;
    }

    [[nodiscard]] bool has_new() const noexcept {
        return region_.as<TalosShm>()->header.write_seq.load(std::memory_order_acquire)
            != last_seq_;
    }

    [[nodiscard]] uint64_t write_seq() const noexcept {
        // 只读计数器，relaxed 足够
        return region_.as<TalosShm>()->header.write_seq.load(std::memory_order_relaxed);
    }

private:
    void copy_consistent_snapshot(const TalosShm& shm) noexcept {
        while (true) {
            const uint64_t begin = shm.header.data_seq.load(std::memory_order_acquire);
            if (begin & 1ULL) {
                continue;
            }

            std::memcpy(&snapshot_, &shm.data, sizeof(snapshot_));

            std::atomic_thread_fence(std::memory_order_acquire);
            const uint64_t end = shm.header.data_seq.load(std::memory_order_acquire);

            if (begin == end && !(end & 1ULL)) {
                return;
            }
        }
    }

    explicit TalosDataReader(ShmRegion&& region) noexcept
        : region_(std::move(region))
        , last_seq_(0) {}

    ShmRegion region_;
    uint64_t last_seq_;
    chrial::TalosData snapshot_{};
};

} // namespace talos::chiral::ipc
