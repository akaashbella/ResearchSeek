#include "vector_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace researchseek {
namespace {

double squared_norm(const float* vector, std::size_t dimension) {
    double norm = 0.0;
    for (std::size_t column = 0; column < dimension; ++column) {
        const double value = vector[column];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Vectors must contain only finite values");
        }
        norm += value * value;
    }
    if (norm == 0.0 || !std::isfinite(norm)) {
        throw std::invalid_argument("Vectors must have finite, nonzero norms");
    }
    return norm;
}

struct Better {
    bool operator()(const Match& left, const Match& right) const {
        return left.score > right.score ||
               (left.score == right.score && left.chunk_id < right.chunk_id);
    }
};

}  // namespace

void validate_inputs(const float* matrix, const float* query, const std::int64_t* ids,
                     std::size_t rows, std::size_t dimension, std::size_t k) {
    if (dimension == 0 || k == 0) {
        throw std::invalid_argument("Dimension and k must be positive");
    }
    const auto max_bytes = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (dimension > max_bytes / sizeof(float) ||
        rows > max_bytes / sizeof(std::int64_t) ||
        rows > (max_bytes / sizeof(float)) / dimension) {
        throw std::invalid_argument("Input dimensions exceed addressable buffer size");
    }
    if (query == nullptr || (rows > 0 && (matrix == nullptr || ids == nullptr))) {
        throw std::invalid_argument("Input buffers must not be null");
    }
    squared_norm(query, dimension);
    std::unordered_set<std::int64_t> seen;
    seen.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        if (!seen.insert(ids[row]).second) {
            throw std::invalid_argument("Chunk IDs must be unique");
        }
        squared_norm(matrix + row * dimension, dimension);
    }
}

std::vector<Match> cosine_top_k(const float* matrix, const float* query,
                              const std::int64_t* ids, std::size_t rows,
                              std::size_t dimension, std::size_t k) {
    const auto limit = std::min(k, rows);
    std::vector<Match> candidates;
    candidates.reserve(limit);
    // Better as the comparator places the worst retained candidate at the root.
    std::priority_queue<Match, std::vector<Match>, Better> heap(Better{}, std::move(candidates));
    const double query_norm = std::sqrt(squared_norm(query, dimension));
    for (std::size_t row = 0; row < rows; ++row) {
        const float* vector = matrix + row * dimension;
        double dot = 0.0;
        double norm = 0.0;
        for (std::size_t column = 0; column < dimension; ++column) {
            const double value = vector[column];
            dot += value * static_cast<double>(query[column]);
            norm += value * value;
        }
        const double score = dot / (std::sqrt(norm) * query_norm);
        const Match candidate{ids[row], score};
        if (heap.size() < limit) {
            heap.push(candidate);
        } else if (limit > 0 && Better{}(candidate, heap.top())) {
            heap.pop();
            heap.push(candidate);
        }
    }
    std::vector<Match> result;
    result.reserve(limit);
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.begin(), result.end(), Better{});
    return result;
}

}  // namespace researchseek
