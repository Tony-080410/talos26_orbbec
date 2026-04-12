#!/usr/bin/env bash

# =============================================================================
# Talos Cross-Platform Build Script
# =============================================================================
# Supported configurations:
#   macOS  | Apple Clang 17+ | libc++   | any arch
#   Linux  | clang 21+       | libstdc++ (via GCC 13+ toolchain) | x86_64/arm64
#
# Usage:
#   ./build.sh [--debug|--release] [--asan] [--test] [--calib|--playground] [--] [app args...]
#
# Examples:
#   ./build.sh
#   ./build.sh --debug
#   ./build.sh --asan
#   ./build.sh --test --no-run
#   ./build.sh --cross
#   ./build.sh --calib -- --camera 0
#   ./build.sh --no-run
#   ./build.sh --verbose
#   ./build.sh --no-redact-build-output
# =============================================================================

set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
YELLOW='\033[33m'
GREEN='\033[32m'
RED='\033[31m'
BLUE='\033[34m'
DIM='\033[2m'
RESET='\033[0m'

status() {
    local symbol="$1"
    local color="$2"
    local msg="$3"
    echo -e "${color}${symbol} ${msg}${RESET}"
}

info() {
    status "[i]" "$BLUE" "$1"
}

ok() {
    status "[o]" "$GREEN" "$1"
}

warn() {
    status "[!]" "$YELLOW" "$1"
}

die() {
    status "[x]" "$RED" "$1"
    exit 1
}

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
BUILD_TYPE="Release"
BUILD_TARGET="talos"
ENABLE_CALIBRATION="OFF"
ENABLE_ASAN="OFF"
ENABLE_TESTING="OFF"
RUN_AFTER_BUILD="ON"
VERBOSE="OFF"
REDACT_BUILD_OUTPUT="ON"
GENERATOR="Ninja"
RUN_ARGS=()

OS=""
ARCH=""
BUILD_JOBS=""
CC=""
CXX=""
CMAKE_EXTRA_ARGS=()
CMAKE_CONFIGURE_ARGS=()
CMAKE_BUILD_ARGS=()
CMAKE_ENV_VARS=()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

detect_os() {
    local os
    os="$(uname -s)"
    case "$os" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        *)      echo "unknown" ;;
    esac
}

detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64)         echo "x86_64" ;;
        aarch64|arm64)  echo "arm64" ;;
        *)              echo "unknown" ;;
    esac
}

get_first_line() {
    "$@" 2>/dev/null | head -n1 || true
}

get_clang_major_version() {
    local version_str
    version_str="$(get_first_line "$1" --version)"

    # Handles:
    #   Apple clang version 17.0.0 ...
    #   clang version 21.0.0 ...
    #   Ubuntu clang version 21.0.0 ...
    if [[ "$version_str" =~ Apple[[:space:]]clang[[:space:]]version[[:space:]]([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$version_str" =~ clang[[:space:]]version[[:space:]]([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "0"
    fi
}

is_apple_clang() {
    local version_str
    version_str="$(get_first_line "$1" --version)"
    [[ "$version_str" == Apple\ clang* ]]
}

get_gcc_major_version() {
    local version_str
    version_str="$(get_first_line "$1" --version)"
    if [[ "$version_str" =~ ([0-9]+)\.[0-9]+\.[0-9]+ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "0"
    fi
}

num_cpus() {
    case "$OS" in
        macos) sysctl -n hw.ncpu ;;
        linux) nproc ;;
        *) echo 4 ;;
    esac
}

join_by() {
    local IFS="$1"
    shift
    echo "$*"
}

ensure_generator() {
    if ! command_exists ninja; then
        die "Ninja not found. Please install ninja."
    fi
}

# -----------------------------------------------------------------------------
# Toolchain selection
# -----------------------------------------------------------------------------
select_macos_compiler() {
    local clangxx
    local clangc

    if command_exists xcrun; then
        clangxx="$(xcrun --find clang++ 2>/dev/null || true)"
        clangc="$(xcrun --find clang 2>/dev/null || true)"
    else
        clangxx="$(command -v clang++ || true)"
        clangc="$(command -v clang || true)"
    fi

    [[ -n "$clangxx" && -x "$clangxx" ]] || die "clang++ not found. Install Xcode Command Line Tools."
    [[ -n "$clangc"  && -x "$clangc"  ]] || die "clang not found. Install Xcode Command Line Tools."

    local major
    major="$(get_clang_major_version "$clangxx")"

    if ! is_apple_clang "$clangxx"; then
        warn "Detected non-Apple clang on macOS: $(get_first_line "$clangxx" --version)"
        warn "This script officially supports Apple Clang on macOS."
    fi

    (( major >= 17 )) || die "Apple Clang 17+ required on macOS. Found: $(get_first_line "$clangxx" --version)"

    CC="$clangc"
    CXX="$clangxx"

    # On macOS, libc++ is the natural/default stdlib for Apple Clang.
    CMAKE_EXTRA_ARGS+=(
        "-DCMAKE_C_COMPILER=${CC}"
        "-DCMAKE_CXX_COMPILER=${CXX}"
    )
}

select_linux_clang() {
    local candidates=(
        clang++-22
        clang++-21
        clang++
    )

    local cxx_candidate=""
    local cc_candidate=""
    local major="0"

    for c in "${candidates[@]}"; do
        if command_exists "$c"; then
            major="$(get_clang_major_version "$c")"
            if (( major >= 21 )); then
                cxx_candidate="$(command -v "$c")"
                break
            fi
        fi
    done

    [[ -n "$cxx_candidate" ]] || die "clang++ 21+ not found. Please install clang-21 or newer."

    cc_candidate="${cxx_candidate/clang++/clang}"
    [[ -x "$cc_candidate" ]] || cc_candidate="$(command -v clang || true)"
    [[ -n "$cc_candidate" && -x "$cc_candidate" ]] || die "Matching clang not found for ${cxx_candidate}."

    CC="$cc_candidate"
    CXX="$cxx_candidate"
}

select_linux_gcc_toolchain() {
    local gcc_candidates=(
        g++-15
        g++-14
        g++-13
        g++
    )

    local gxx=""
    local gcc_major="0"

    for c in "${gcc_candidates[@]}"; do
        if command_exists "$c"; then
            gcc_major="$(get_gcc_major_version "$c")"
            if (( gcc_major >= 13 )); then
                gxx="$( command -v "$c")"
                break
            fi
        fi
    done

    [[ -n "$gxx" ]] || die "GCC/G++ 13+ toolchain required on Linux for a recent libstdc++."

    local gcc_root
    gcc_root="$(cd "$(dirname "$gxx")/.." && pwd)"

    info "Using GCC toolchain: $gxx"
    info "GCC version: $(get_first_line "$gxx" --version)"

    # Try to locate libstdc++.so as seen by clang / system
    local libstdcpp_path=""
    libstdcpp_path="$("$CXX" -print-file-name=libstdc++.so 2>/dev/null || true)"

    if [[ -z "$libstdcpp_path" || "$libstdcpp_path" == "libstdc++.so" ]]; then
        # fallback via g++
        libstdcpp_path="$("$gxx" -print-file-name=libstdc++.so 2>/dev/null || true)"
    fi

    local libstdcpp_dir=""
    if [[ -n "$libstdcpp_path" && "$libstdcpp_path" != "libstdc++.so" ]]; then
        libstdcpp_dir="$(dirname "$libstdcpp_path")"
        info "libstdc++: $libstdcpp_path"
    else
        warn "Could not resolve libstdc++.so path directly; relying on GCC toolchain discovery."
    fi

    # Make clang use the intended GCC toolchain explicitly.
    CMAKE_EXTRA_ARGS+=(
        "-DCMAKE_C_COMPILER=${CC}"
        "-DCMAKE_CXX_COMPILER=${CXX}"
        "-DCMAKE_C_FLAGS=--gcc-toolchain=${gcc_root}"
        "-DCMAKE_CXX_FLAGS=--gcc-toolchain=${gcc_root} -stdlib=libstdc++"
    )

    if [[ -n "$libstdcpp_dir" ]]; then
        export CMAKE_LIBRARY_PATH="${libstdcpp_dir}:${CMAKE_LIBRARY_PATH:-}"
    fi
}

# -----------------------------------------------------------------------------
# Platform configuration
# -----------------------------------------------------------------------------
configure_macos() {
    info "Configuring for macOS..."
    select_macos_compiler
    BUILD_JOBS="$(num_cpus)"
    ok "macOS ready (Apple Clang + libc++)"
}

configure_linux() {
    info "Configuring for Linux..."
    select_linux_clang
    info "Compiler: $(get_first_line "$CXX" --version)"
    select_linux_gcc_toolchain
    BUILD_JOBS="$(num_cpus)"
    ok "Linux ready ($(basename "$CXX") + libstdc++ via GCC toolchain)"
}

# -----------------------------------------------------------------------------
# CMake / build / run
# -----------------------------------------------------------------------------
configure_cmake() {
    CMAKE_CONFIGURE_ARGS=(
        -G "$GENERATOR"
        -B build
        -S .
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DENABLE_CALIBRATION=${ENABLE_CALIBRATION}"
        "-DENABLE_ASAN=${ENABLE_ASAN}"
        "-DTALOS_BUILD_TESTING=${ENABLE_TESTING}"
        "${CMAKE_EXTRA_ARGS[@]}"
    )

    info "Configuring CMake..."
    cmake "${CMAKE_CONFIGURE_ARGS[@]}"
}

build_target() {
    export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"

    CMAKE_BUILD_ARGS=(
        --build build
    )

    if [[ "$ENABLE_TESTING" == "OFF" ]]; then
        CMAKE_BUILD_ARGS+=(--target "${BUILD_TARGET}")
    fi

    if [[ "$VERBOSE" == "ON" ]]; then
        CMAKE_BUILD_ARGS+=(--verbose)
    fi

    if [[ "$ENABLE_TESTING" == "ON" ]]; then
        info "Building all targets (${BUILD_TYPE})..."
    else
        info "Building ${BUILD_TARGET} (${BUILD_TYPE})..."
    fi
    cmake "${CMAKE_BUILD_ARGS[@]}"

    ok "Build complete!"
}

run_target() {
    [[ "$RUN_AFTER_BUILD" == "ON" ]] || return 0

    local exe="./build/bin/${BUILD_TARGET}"
    [[ -x "$exe" ]] || die "Built executable not found: $exe"

    info "Starting ${BUILD_TARGET}..."
    if [[ "${#RUN_ARGS[@]}" -gt 0 ]]; then
        info "App args: $(join_by ' ' "${RUN_ARGS[@]}")"
    fi

    exec "$exe"
}

run_tests() {
    [[ "$ENABLE_TESTING" == "ON" ]] || return 0

    info "Running tests..."
    ctest --test-dir build --output-on-failure --parallel "${BUILD_JOBS}"
    ok "All tests passed!"
}

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --release)
                BUILD_TYPE="Release"
                shift
                ;;
            --debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            --asan|-fsanitize=address)
                ENABLE_ASAN="ON"
                shift
                ;;
            --test)
                ENABLE_TESTING="ON"
                RUN_AFTER_BUILD="OFF"
                shift
                ;;
            -c|--calib)
                BUILD_TARGET="rm_calibration"
                ENABLE_CALIBRATION="ON"
                shift
                ;;
            -cx|--cross)
                CMAKE_EXTRA_ARGS+=(
                    "-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu-clang.cmake"
                        "-DTALOS_TARGET_TRIPLE=aarch64-unknown-linux-gnu"
                        "-DTALOS_LLVM_ROOT=$(brew --prefix llvm)"
                        "-DTALOS_GNU_TOOLCHAIN_ROOT=$(brew --prefix aarch64-unknown-linux-gnu)/toolchain"
                        "-DTALOS_SYSROOT=/Users/blackjack/sysroots/ax650n"
                        "-DBUILD_TESTING=OFF"
                        "-DCMAKE_CROSSCOMPILING=ON"
                )
                shift
                ;;
            -p|--playground|--play)
                BUILD_TARGET="talos-playground"
                shift
                ;;
            --no-run)
                RUN_AFTER_BUILD="OFF"
                shift
                ;;
            --verbose)
                VERBOSE="ON"
                shift
                ;;
            --no-redact-build-output)
                REDACT_BUILD_OUTPUT="OFF"
                shift
                ;;
            --redact-build-output)
                REDACT_BUILD_OUTPUT="ON"
                shift
                ;;
            --)
                shift
                RUN_ARGS+=("$@")
                break
                ;;
            *)
                RUN_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"

    echo ""
    status "🤖" "$BLUE" "Talos Build & Run"
    echo ""

    OS="$(detect_os)"
    ARCH="$(detect_arch)"

    [[ "$OS"   != "unknown" ]] || die "Unsupported OS: $(uname -s)"
    [[ "$ARCH" != "unknown" ]] || die "Unsupported architecture: $(uname -m)"

    ensure_generator

    info "Platform: ${OS} (${ARCH})"
    info "Build type: ${BUILD_TYPE}"
    info "ASAN: ${ENABLE_ASAN}"
    info "Target: ${BUILD_TARGET}"
    info "Run after build: ${RUN_AFTER_BUILD}"
    info "Build log redaction: ${REDACT_BUILD_OUTPUT}"

    case "$OS" in
        macos) configure_macos ;;
        linux) configure_linux ;;
        *) die "Unsupported platform" ;;
    esac

    info "CC  = ${CC}"
    info "CXX = ${CXX}"
    info "Jobs = ${BUILD_JOBS}"

    configure_cmake
    build_target
    run_tests
    run_target
}

main "$@"
