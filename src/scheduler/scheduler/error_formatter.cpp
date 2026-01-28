#include "scheduler/scheduler/error_formatter.hpp"

namespace talos {

auto format_build_error(const BuildError& err) -> std::string {
    return std::visit(
        [](const auto& e) -> std::string {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, SchedulerError>) {
                return std::string(to_string(e));
            } else if constexpr (std::is_same_v<T, ChannelKindConflict>) {
                return "ChannelKindConflict: channel used as both SPSC and SPMC by '"
                     + e.first_system + "' and '" + e.second_system + "'";
            } else if constexpr (std::is_same_v<T, MultipleWritersError>) {
                std::string s = "MultipleWriters: channel has multiple writers: ";
                for (std::size_t i = 0; i < e.writers.size(); ++i) {
                    if (i > 0)
                        s += ", ";
                    s += "'" + e.writers[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, MultipleReadersError>) {
                std::string s = "MultipleReaders: SPSC channel has multiple readers: ";
                for (std::size_t i = 0; i < e.readers.size(); ++i) {
                    if (i > 0)
                        s += ", ";
                    s += "'" + e.readers[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, OrphanedReaderError>) {
                std::string s = "OrphanedReader: reader(s) without writer: ";
                for (std::size_t i = 0; i < e.readers.size(); ++i) {
                    if (i > 0)
                        s += ", ";
                    s += "'" + e.readers[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, DependencyCycleError>) {
                std::string s = "DependencyCycle: ";
                for (std::size_t i = 0; i < e.cycle.size(); ++i) {
                    if (i > 0)
                        s += " -> ";
                    s += "'" + e.cycle[i] + "'";
                }
                return s;
            } else if constexpr (std::is_same_v<T, TooManyComputeSystemsError>) {
                return "TooManyComputeSystems: " + std::to_string(e.count) + " (max "
                     + std::to_string(e.max_count) + ")";
            }
            return "Unknown error";
        },
        err);
}

} // namespace talos
