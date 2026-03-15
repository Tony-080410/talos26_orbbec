#!/usr/bin/env bash

# =============================================================================
# Talos Cross-Platform Build Script
# =============================================================================
# Supported configurations:
#   macOS     | Apple Clang 17+ | libSystem   | any arch
#   Linux     | clang-21        | libstdc++ >= 23 | any arch
#
# Usage:
#   ./build.sh [--release] [--asan] [--calib] [--] [args...]
#     --release       Build in Release mode (default: Debug)
#     --asan          Enable AddressSanitizer (memory leak detection)
#     --calib         Build calibration app (default: main talos app)
#     --             Separator for app arguments
# =============================================================================

set -euo pipefail

# Colors
YELLOW='\033[33m'
GREEN='\033[32m'
RED='\033[31m'
BLUE='\033[34m'
RESET='\033[0m'

status() {
    local symbol="$1"
    local color="$2"
    local msg="$3"
    echo -e "${color}${symbol} ${msg}${RESET}"
}

error() {
    status "[x]" "$RED" "$1"
    exit 1
}

# Detect OS
detect_os() {
    local os
    os="$(uname -s)"
    case "$os" in
        Darwin)
            echo "macos"
            ;;
        Linux)
            echo "linux"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Detect architecture
detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|aarch64|arm64)
            echo "$arch"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Check compiler availability
check_compiler() {
    command -v "$1" &> /dev/null
}

# Get compiler version
get_compiler_version() {
    "$1" --version 2>/dev/null | head -n1 || echo "unknown"
}

# Validate libstdc++ version on Linux
check_libstdcpp_version() {
    local required_major="$1"
    local output
    local version_major

    if command -v g++ &> /dev/null; then
        output="$(g++ -v 2>&1 | grep "libstdc++" || true)"
        if [[ "$output" =~ libstdc\+\+[\ \(]*([0-9]+) ]]; then
            version_major="${BASH_REMATCH[1]}"
            if [[ "$version_major" -ge "$required_major" ]]; then
                return 0
            fi
        fi
    fi

    local libstdcpp_path
    libstdcpp_path="$(ldd $(command -v ls) 2>/dev/null | grep libstdc++ | awk '{print $3}' | head -n1 || true)"
    if [[ -n "$libstdcpp_path" && -f "$libstdcpp_path" ]]; then
        local version_str
        version_str="$("$libstdcpp_path" --version 2>/dev/null | head -n1 || true)"
        if [[ "$version_str" =~ libstdc\+\+.*([0-9]+)\.([0-9]+) ]]; then
            local major="${BASH_REMATCH[1]}"
            if [[ "$major" -ge "$required_major" ]]; then
                return 0
            fi
        fi
    fi

    return 1
}

# Configure build environment for macOS
configure_macos() {
    status "[+]" "$BLUE" "Configuring for macOS..."

    if ! check_compiler "clang++"; then
        error "clang++ not found. Install Xcode Command Line Tools."
    fi

    local clang_version
    clang_version="$(get_compiler_version "clang++")"
    status "[i]" "$BLUE" "Compiler: $clang_version"

    # macOS uses libSystem by default
    export CC=cc
    export CXX=c++

    # Build flags for macOS
    CMAKE_COMPILER_ARGS=""
    BUILD_JOBS="$(sysctl -n hw.ncpu)"

    status "[o]" "$GREEN" "macOS ready (Apple Clang + libSystem)"
}

# Find latest libstdc++ library path
find_latest_libstdcpp() {
    local lib_dir
    local latest_lib=""
    local latest_ver="0"

    # Common library directories
    for lib_dir in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib; do
        if [[ ! -d "$lib_dir" ]]; then
            continue
        fi

        # Find all libstdc++.so.* files and extract version numbers
        while IFS= read -r lib; do
            if [[ "$lib" =~ libstdc\+\+\.so\.([0-9]+)\.([0-9]+) ]]; then
                local major="${BASH_REMATCH[1]}"
                local minor="${BASH_REMATCH[2]}"
                local ver_num="$((major * 1000 + minor))"

                if [[ "$ver_num" -gt "$latest_ver" ]]; then
                    latest_ver="$ver_num"
                    latest_lib="$lib"
                fi
            fi
        done < <(find "$lib_dir" -maxdepth 1 -name "libstdc++.so.*" 2>/dev/null)
    done

    if [[ -n "$latest_lib" ]]; then
        echo "$latest_lib"
    fi
}

# Check compiler version
get_compiler_major_version() {
    local version_str
    version_str="$("$1" --version 2>/dev/null | head -n1 || echo "")"

    # Extract version number (handles formats like "clang version 21.0.0" or "Ubuntu clang version 21.0.0")
    if [[ "$version_str" =~ clang[[:space:]]version[[:space:]]([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$version_str" =~ version[[:space:]]([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "0"
    fi
}

# Configure build environment for Linux
configure_linux() {
    status "[+]" "$BLUE" "Configuring for Linux..."

    local selected_cc="clang-21"
    local selected_cxx="clang++-21"
    local clang_version

    # Try clang-21 first
    if check_compiler "clang++-21"; then
        clang_version="$(get_compiler_version "clang++-21")"
        status "[i]" "$BLUE" "Compiler: $clang_version"
    else
        # Fallback to clang/clang++ and check version
        if check_compiler "clang++"; then
            local major_version
            major_version="$(get_compiler_major_version "clang++")"

            if [[ "$major_version" -ge 21 ]]; then
                selected_cc="clang"
                selected_cxx="clang++"
                clang_version="$(get_compiler_version "clang++")"
                status "[i]" "$BLUE" "Compiler: $clang_version"
            else
                status "[x]" "$RED" "clang++ version $major_version is too old (requires 21+)"
                status "[i]" "$YELLOW" "Install: sudo apt install clang-21 clang++-21"
                error "Required compiler version not available."
            fi
        else
            status "[x]" "$RED" "clang++-21 not found and clang++ not available!"
            status "[i]" "$YELLOW" "Install: sudo apt install clang-21 clang++-21"
            error "Required compiler not available."
        fi
    fi

    # Find latest libstdc++
    local latest_libstdcpp
    latest_libstdcpp="$(find_latest_libstdcpp)"

    if [[ -z "$latest_libstdcpp" ]]; then
        status "[x]" "$RED" "libstdc++ not found in system paths!"
        error "Required stdlib not available."
    fi

    local libstdcpp_dir
    libstdcpp_dir="$(dirname "$latest_libstdcpp")"
    status "[i]" "$BLUE" "Latest libstdc++: $latest_libstdcpp"

    # Get version info
    local libstdcpp_version
    libstdcpp_version="$("$latest_libstdcpp" --version 2>/dev/null | head -n1 || echo "unknown")"
    status "[i]" "$BLUE" "Stdlib: $libstdcpp_version"

    export CC="$selected_cc"
    export CXX="$selected_cxx"

    # Explicit compiler flags for Linux - force latest libstdc++
    CMAKE_COMPILER_ARGS="-DCMAKE_CXX_COMPILER=$CXX -DCMAKE_C_COMPILER=$CC"
    CMAKE_LIBRARY_PATH="$libstdcpp_dir:${CMAKE_LIBRARY_PATH:-}"
    export CMAKE_LIBRARY_PATH

    # Add rpath to ensure runtime uses the correct libstdc++
    CMAKE_CXX_FLAGS="-stdlib=libstdc++"

    BUILD_JOBS="$(nproc)"

    status "[o]" "$GREEN" "Linux ready ($selected_cxx + latest libstdc++)"
}

# Main
main() {
    # Parse arguments
    BUILD_TYPE="Release"
    BUILD_TARGET="talos"
    ENABLE_CALIBRATION="OFF"
    ENABLE_ASAN="OFF"
    RUN_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --release)
                BUILD_TYPE="Release"
                shift
                ;;
            --asan|-fsanitize=address)
                ENABLE_ASAN="ON"
                shift
                ;;
            -c|--calib)
                BUILD_TARGET="rm_calibration"
                ENABLE_CALIBRATION="ON"
                shift
                ;;
            -p|--playground|--play)
                BUILD_TARGET="talos-playground"
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

    echo ""
    status "🤖" "$BLUE" "Talos Build & Run"
    echo ""

    # Detect platform
    OS="$(detect_os)"
    ARCH="$(detect_arch)"

    if [[ "$OS" == "unknown" ]]; then
        error "Unsupported OS: $(uname -s)"
    fi

    if [[ "$ARCH" == "unknown" ]]; then
        error "Unsupported architecture: $(uname -m)"
    fi

    status "[i]" "$BLUE" "Platform: $OS ($ARCH)"
    status "[i]" "$BLUE" "Build: $BUILD_TYPE"
    status "[i]" "$BLUE" "ASAN: $ENABLE_ASAN"
    status "[i]" "$BLUE" "Target: $BUILD_TARGET"

    # Configure based on OS
    case "$OS" in
        macos)
            configure_macos
            ;;
        linux)
            configure_linux
            ;;
        *)
            error "Unsupported platform"
            ;;
    esac

    status "[i]" "$BLUE" "CC  = $CC"
    status "[i]" "$BLUE" "CXX = $CXX"

    # Configure CMake
    status "[>]" "$GREEN" "Configuring CMake..."
    cmake -G Ninja -B build -S . \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DENABLE_CALIBRATION="${ENABLE_CALIBRATION}" \
        -DENABLE_ASAN="${ENABLE_ASAN}" \
        ${CMAKE_COMPILER_ARGS} \
        ${CMAKE_CXX_FLAGS:+-DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS"}

    # Build
    status "[>]" "$GREEN" "Building $BUILD_TARGET ($BUILD_TYPE)..."
    cmake --build build -- "$BUILD_TARGET" -j"${BUILD_JOBS}"

    status "[o]" "$GREEN" "Build complete!"

    # Platform-specific run setup
    case "$OS" in
        linux)
            # Set HIK SDK library path for Linux
            HIK_LIB="$(pwd)/3dparty/hik_sdk/lib/arm64"
            if [[ -d "$HIK_LIB" ]]; then
                export LD_LIBRARY_PATH="$HIK_LIB:$LD_LIBRARY_PATH"
                status "[i]" "$BLUE" "LD_LIBRARY_PATH=$HIK_LIB"
            fi
            ;;
    esac

    # Run
    status "[>]" "$GREEN" "Starting $BUILD_TARGET..."
    exec "./build/bin/$BUILD_TARGET"
}

main "$@"
