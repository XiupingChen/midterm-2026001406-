#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

std::vector<double> generateNormal(int n, double mean, double stddev) {
    if (n < 2 || !std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0.0) {
        throw std::invalid_argument("Use n >= 2, a finite mean and a positive finite SD.");
    }
    std::default_random_engine generator(2026);
    std::normal_distribution<double> distribution(mean, stddev);
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) result[i] = distribution(generator);
    return result;
}

// Provided merge helper, operating on the half-open ranges [left, mid), [mid, right).
void mergeRange(std::vector<double>& values, std::size_t left,
                 std::size_t mid, std::size_t right) {
    std::vector<double> merged;
    std::size_t i = left, j = mid;
    while (i < mid && j < right) {
        merged.push_back(values[i] <= values[j] ? values[i++] : values[j++]);
    }
    while (i < mid) merged.push_back(values[i++]);
    while (j < right) merged.push_back(values[j++]);
    for (std::size_t k = 0; k < merged.size(); ++k) values[left + k] = merged[k];
}

class RandomNormalVector {
private:
    std::vector<double> Vec;
    static void mergeSort(std::vector<double>& values, std::size_t left,
                           std::size_t right) {
        if (right - left <= 1) return;
        const std::size_t mid = left + (right - left) / 2;
        mergeSort(values, left, mid);
        mergeSort(values, mid, right);
        mergeRange(values, left, mid, right);
    }
public:
    explicit RandomNormalVector(int n) : Vec(generateNormal(n, 0.0, 1.0)) {}
    RandomNormalVector(int n, double mean, double stddev)
        : Vec(generateNormal(n, mean, stddev)) {}
    // Provided only for deterministic tests.
    explicit RandomNormalVector(const std::vector<double>& values) : Vec(values) {
        if (Vec.size() < 2) throw std::invalid_argument("At least two values required.");
    }
    const std::vector<double>& values() const { return Vec; }
    void sort() { mergeSort(Vec, 0, Vec.size()); }
    double avg() const {
        // Provided mean helper; you may use it in sd().
        double sum = 0.0;
        for (double value : Vec) sum += value;
        return sum / Vec.size();
    }
    double sd() const {
        const double mean = avg();
        double sumSquares = 0.0;
        for (double value : Vec) {
            const double difference = value - mean;
            sumSquares += difference * difference;
        }
        return std::sqrt(sumSquares / (Vec.size() - 1));
    }
    friend std::ostream& operator<<(std::ostream& os, const RandomNormalVector& v);
};

std::ostream& operator<<(std::ostream& os, const RandomNormalVector& v) {
    for (std::size_t i = 0; i < v.Vec.size(); ++i) {
        if (i != 0) os << ' ';
        os << v.Vec[i];
    }
    return os;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    int failures = 0;
    auto run = [&](const char* name, const std::function<void()>& test) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
        }
    };
    run("parameterized constructor", [] {
        RandomNormalVector v(10, 2.0, 1.5);
        const auto expected = generateNormal(10, 2.0, 1.5);
        require(v.values() == expected, "Check constructor parameters and stored values.");
    });
    run("merge sort", [] {
        RandomNormalVector v(std::vector<double>{3, -1, 5, -1, 2});
        v.sort();
        require(v.values() == std::vector<double>({-1, -1, 2, 3, 5}), "Incorrect sort.");
    });
    run("mean", [] {
        RandomNormalVector v(std::vector<double>{1, 2, 3, 4, 5});
        require(std::abs(v.avg() - 3.0) < 1e-12, "Incorrect mean.");
    });
    run("sample standard deviation", [] {
        RandomNormalVector v(std::vector<double>{1, 2, 3, 4, 5});
        require(std::abs(v.sd() - std::sqrt(2.5)) < 1e-12, "Incorrect sample SD.");
    });
    run("stream output", [] {
        RandomNormalVector v(std::vector<double>{1, 2, 3});
        std::ostringstream output;
        output << v;
        std::istringstream input(output.str());
        double a, b, c;
        require(bool(input >> a >> b >> c) && a == 1 && b == 2 && c == 3,
                "Output three space-separated numbers.");
    });
    run("Normal examples", [] {
        RandomNormalVector v1(10), v2(10, 2.0, 1.5);
        std::cout << "v1: " << v1 << '\n';
        v1.sort();
        std::cout << "v1 sorted: " << v1 << '\n';
        std::cout << "v2: " << v2 << '\n';
        std::cout << "v2 mean: " << v2.avg() << ", SD: " << v2.sd() << '\n';
    });
    return failures == 0 ? 0 : 1;
}
