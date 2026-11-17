#include "vector_search.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

int main() {
    const std::size_t rows = 257;
    const std::size_t dimension = 17;
    std::mt19937 generator(23);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    std::vector<float> matrix(rows * dimension);
    std::vector<float> query(dimension);
    std::vector<std::int64_t> ids(rows);
    for (auto& value : matrix) value = distribution(generator);
    for (auto& value : query) value = distribution(generator);
    for (std::size_t row = 0; row < rows; ++row) ids[row] = static_cast<std::int64_t>(rows - row);
    for (std::size_t k : {1, 9, 257, 300}) {
        researchseek::validate_inputs(matrix.data(), query.data(), ids.data(), rows, dimension, k);
        const auto matches = researchseek::cosine_top_k(matrix.data(), query.data(), ids.data(), rows, dimension, k);
        assert(matches.size() == std::min(k, rows));
        for (std::size_t index = 0; index < matches.size(); ++index) {
            assert(std::isfinite(matches[index].score));
            if (index > 0) assert(matches[index - 1].score >= matches[index].score);
        }
    }
    researchseek::validate_inputs(nullptr, query.data(), nullptr, 0, dimension, 1);
    assert(researchseek::cosine_top_k(nullptr, query.data(), nullptr, 0, dimension, 1).empty());
    auto expect_invalid = [&](std::size_t n, std::size_t d, std::size_t k) {
        bool caught = false;
        try {
            researchseek::validate_inputs(matrix.data(), query.data(), ids.data(), n, d, k);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        assert(caught);
    };
    expect_invalid(rows, 0, 1);
    expect_invalid(rows, dimension, 0);
    expect_invalid(std::numeric_limits<std::size_t>::max(), dimension, 1);
    expect_invalid(rows, std::numeric_limits<std::size_t>::max(), 1);
    ids[0] = ids[1];
    expect_invalid(rows, dimension, 1);
    ids[0] = 999;
    matrix[0] = std::numeric_limits<float>::infinity();
    expect_invalid(rows, dimension, 1);
    matrix[0] = 1;
    std::fill(query.begin(), query.end(), 0.0f);
    expect_invalid(rows, dimension, 1);
    std::cout << "Native safety checks passed\n";
}
