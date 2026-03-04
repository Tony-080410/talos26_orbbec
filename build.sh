#!/bin/zsh

YELLOW="\033[33m"
GREEN="\033[32m"
RED="\033[31m"
BLUE="\033[34m"
RESET="\033[0m"

status() {
    local symbol="$1"
    local color="$2"
    local msg="$3"
    echo -e "${color}${symbol} ${msg}${RESET}"
}

download_file() {
    local target_dir="$1"
    local url="$2"
    local filename="$(basename "$url")"
    local dest="$target_dir/$filename"

    mkdir -p "$target_dir"

    if [[ -f "$dest" ]]; then
        status "[x]" "$RED" "$dest already exists, skipping download."
        return 0
    fi

    status "[!]" "$YELLOW" "Downloading $filename..."
    if wget -O "$dest" "$url"; then
        status "[o]" "$GREEN" "Downloaded $filename successfully."
    else
        status "[x]" "$RED" "Failed to download $filename."
        return 1
    fi
}

clean_dir() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        status "[!]" "$YELLOW" "Removing existing $dir..."
        rm -rf "$dir"
        status "[o]" "$GREEN" "Removed $dir."
    else
        status "[x]" "$RED" "No existing directory."
    fi
}

BUILD_TYPE="${BUILD_TYPE:-Release}"

cmake -G Ninja -B build -S . \
 -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
 -DENABLE_CALIBRATION=OFF
# -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
# -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build build -- talos -j64
status "[+]" "$BLUE" "All done."
