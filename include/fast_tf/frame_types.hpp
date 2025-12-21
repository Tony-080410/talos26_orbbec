#pragma once

#include <string_view>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <string>
#include "compile_time_string.hpp"
#include "concepts.hpp"

namespace fast_tf {

// ============================================================================
// 编译期坐标系ID定义（注：更好的实现在 frame_builder.hpp 中）
// ============================================================================

// 旧的实现已移至 frame_builder.hpp，新项目应使用 frame_builder.hpp 中的 FrameId

// 宏定义，用于简化编译期帧ID的创建
#define DECLARE_FRAME_ID(frame_name_str) \
    struct frame_name##Frame { \
        static constexpr std::string_view name() { \
            return std::string_view(#frame_name_str); \
        } \
        static constexpr uint64_t id() { \
            return fast_tf::hash_string_view(std::string_view(#frame_name_str)); \
        } \
    }

// ============================================================================
// 运行期帧名称和ID管理
// ============================================================================

class FrameRegistry {
public:
    // 注册新的坐标系及其别名
    static void register_frame(std::string_view name, uint64_t frame_id) {
        if (auto it = frame_map.find(frame_id); it == frame_map.end()) {
            frame_map[frame_id] = std::string(name);
        }
    }

    // 注册别名
    static void register_alias(std::string_view alias, std::string_view canonical_name) {
        if (auto it = canonical_map.find(std::string(alias)); it == canonical_map.end()) {
            canonical_map[std::string(alias)] = std::string(canonical_name);
        }
    }

    // 根据别名获取规范名称
    static std::string_view resolve_alias(std::string_view alias) {
        if (auto it = canonical_map.find(std::string(alias));
            it != canonical_map.end()) {
            return it->second;
        }
        return alias;
    }

    // 根据ID获取帧名称
    static std::string_view get_frame_name(uint64_t frame_id) {
        if (auto it = frame_map.find(frame_id); it != frame_map.end()) {
            return it->second;
        }
        return "unknown_frame";
    }

    // 根据名称获取帧ID
    static uint64_t get_frame_id(std::string_view name) {
        std::string canonical = std::string(resolve_alias(name));
        for (const auto& [id, name_str] : frame_map) {
            if (name_str == canonical) {
                return id;
            }
        }
        // 如果未找到，计算哈希值
        return hash_string_view(name);
    }

    // 清空注册表（用于测试）
    static void clear() {
        frame_map.clear();
        canonical_map.clear();
    }

private:
    static std::unordered_map<uint64_t, std::string> frame_map;
    static std::unordered_map<std::string, std::string> canonical_map;
};

// ============================================================================
// 变换输入类型
// ============================================================================

// RPY格式的变换
struct RPYTransformData {
    double roll;   // 绕X轴旋转（弧度）
    double pitch;  // 绕Y轴旋转（弧度）
    double yaw;    // 绕Z轴旋转（弧度）
    double x;      // 平移X
    double y;      // 平移Y
    double z;      // 平移Z

    RPYTransformData() = default;

    RPYTransformData(double r, double p, double y,
                     double tx, double ty, double tz)
        : roll(r), pitch(p), yaw(y), x(tx), y(ty), z(tz) {}

    // 验证值有效性
    [[nodiscard]] bool is_valid() const {
        // 在 -ffinite-math-only 模式下，我们跳过显式的 isfinite 检查
        // 这里仅做基本的范围检查
        return true;  // 在 -ffinite-math-only 下，所有数值都被认为是有限的
    }
};

// 四元数格式的变换
struct QuaternionTransformData {
    double w;   // 实部
    double x;   // 虚部i
    double y;   // 虚部j
    double z;   // 虚部k
    double tx;  // 平移X
    double ty;  // 平移Y
    double tz;  // 平移Z

    QuaternionTransformData() = default;

    QuaternionTransformData(double w_val, double x_val, double y_val,
                            double z_val, double tx_val, double ty_val,
                            double tz_val)
        : w(w_val), x(x_val), y(y_val), z(z_val),
          tx(tx_val), ty(ty_val), tz(tz_val) {}

    // 验证值有效性（检查是否为单位四元数）
    [[nodiscard]] bool is_valid() const {
        // 在 -ffinite-math-only 模式下，跳过 isfinite 检查
        // 检查四元数范数（应接近1.0）
        double norm = w * w + x * x + y * y + z * z;
        return std::abs(norm - 1.0) < 1e-6;
    }

    // 归一化四元数
    void normalize() {
        double norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (norm > 1e-10) {
            w /= norm;
            x /= norm;
            y /= norm;
            z /= norm;
        }
    }
};

// ============================================================================
// 变换矩阵包装器
// ============================================================================

class TransformMatrix4d {
public:
    using Matrix = Eigen::Matrix4d;

    TransformMatrix4d() : matrix_(Matrix::Identity()) {}

    explicit TransformMatrix4d(const Matrix& m) : matrix_(m) {}

    // 从RPY和平移创建
    static TransformMatrix4d from_rpy(double roll, double pitch, double yaw,
                                      double x, double y, double z) {
        //todo check wheter eigen provide this method              
        Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q = rz * ry * rx;

        Matrix m = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m(0, 3) = x;
        m(1, 3) = y;
        m(2, 3) = z;

        return TransformMatrix4d(m);
    }

    // 从四元数和平移创建
    static TransformMatrix4d from_quaternion(double w, double x, double y,
                                             double z, double tx, double ty,
                                             double tz) {
                                                
        Eigen::Quaterniond q(w, x, y, z);
        q.normalize();
        //todo check wheter eigen provide this method              
        Matrix m = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m(0, 3) = tx;
        m(1, 3) = ty;
        m(2, 3) = tz;

        return TransformMatrix4d(m);
    }

    // 矩阵乘法
    TransformMatrix4d operator*(const TransformMatrix4d& other) const {
        return TransformMatrix4d(matrix_ * other.matrix_);
    }

    // 获取逆变换
    [[nodiscard]] TransformMatrix4d inverse() const {
        return TransformMatrix4d(matrix_.inverse());
    }

    // 获取底层Eigen矩阵
    [[nodiscard]] const Matrix& matrix() const { return matrix_; }
    [[nodiscard]] Matrix& matrix() { return matrix_; }

    // 获取平移向量
    [[nodiscard]] Eigen::Vector3d translation() const {
        return matrix_.block<3, 1>(0, 3);
    }

    // 获取旋转矩阵
    [[nodiscard]] Eigen::Matrix3d rotation() const {
        return matrix_.block<3, 3>(0, 0);
    }

    // 获取四元数
    [[nodiscard]] Eigen::Quaterniond quaternion() const {
        return Eigen::Quaterniond(rotation());
    }

    // 线性插值
    static TransformMatrix4d lerp(const TransformMatrix4d& a,
                                  const TransformMatrix4d& b, double t) {
        // 位置线性插值
        Eigen::Vector3d pos = a.translation() * (1.0 - t) + b.translation() * t;

        // 四元数球面线性插值
        Eigen::Quaterniond qa(a.rotation());
        Eigen::Quaterniond qb(b.rotation());
        Eigen::Quaterniond q = qa.slerp(t, qb);

        Matrix m = Matrix::Identity();
        m.block<3, 3>(0, 0) = q.toRotationMatrix();
        m.block<3, 1>(0, 3) = pos;

        return TransformMatrix4d(m);
    }

private:
    Matrix matrix_;
};

} // namespace fast_tf

// ============================================================================
// 标准库支持
// ============================================================================

namespace std {

template<std::size_t N>
struct hash<fast_tf::CompileTimeString<N>> {
    std::size_t operator()(const fast_tf::CompileTimeString<N>& str) const {
        return static_cast<std::size_t>(str.hash());
    }
};

template<>
struct hash<fast_tf::TransformMatrix4d> {
    std::size_t operator()(const fast_tf::TransformMatrix4d& tm) const {
        // 使用矩阵第一个元素的哈希作为简单实现
        return hash<double>()(tm.matrix()(0, 0));
    }
};

} // namespace std
