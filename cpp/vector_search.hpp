#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace researchseek {

struct Match {
    std::int64_t chunk_id;
    double score;
};

void validate_inputs(const float* matrix, const float* query, const std::int64_t* ids,
                     std::size_t rows, std::size_t dimension, std::size_t k);

// Inputs must already be validated and remain unchanged throughout this call.
std::vector<Match> cosine_top_k(const float* matrix, const float* query,
                              const std::int64_t* ids, std::size_t rows,
                              std::size_t dimension, std::size_t k);

}  // namespace researchseek
