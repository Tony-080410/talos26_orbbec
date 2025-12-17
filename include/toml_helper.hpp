#pragma once

#define READ(tbl, key, field)                                                                                          \
    if (auto val = (tbl)[key].value<decltype(field)>(); val)                                                           \
        field = *val;                                                                                                  \
    else                                                                                                               \
        return std::unexpected(#key " not set or invalid")
