#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <Eigen/Dense>

#include "MyLinAlg/Matrix.hpp"
#include "MyLinAlg/Decompositions.hpp"

using namespace std::chrono;
using std::size_t;

template <typename F>
static long long time_function_ns(F&& f, int iters) {
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (iters > 0 ? iters : 1);
}

static void fill_random(double* data, size_t n, unsigned seed) {
    // Simple LCG for reproducibility and no <random> overhead
    unsigned x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = 1664525u * x + 1013904223u;
        data[i] = (x & 0xFFFFu) / 65535.0 - 0.5; // [-0.5, 0.5]
    }
}

struct BenchPrinter {
    bool csv_output;
    bool header_printed;

    explicit BenchPrinter(bool csv)
        : csv_output(csv), header_printed(false) {}

    void print_header() {
        if (header_printed) return;
        header_printed = true;
        if (csv_output) {
            std::cout << "op,size,avg_ns" << '\n';
        } else {
            // Pretty table header
            std::cout << std::left << std::setw(15) << "Operation"
                      << std::right << std::setw(12)  << "Size"
                      << std::right << std::setw(20) << "Avg (ns)"
                      << std::right << std::setw(18) << "Avg (ms)" << '\n';
            std::cout << std::string(12 + 8 + 16 + 14, '-') << '\n';
        }
    }

    void print_row(const std::string& op, int size, long long avg_ns) {
        if (csv_output) {
            std::cout << op << "," << size << "," << avg_ns << '\n';
        } else {
            const double avg_ms = static_cast<double>(avg_ns) / 1e6;
            std::ostringstream ms_s;
            ms_s.setf(std::ios::fixed);
            ms_s << std::setprecision(3) << avg_ms;

            std::cout << std::left  << std::setw(15) << op
                      << std::right << std::setw(12)  << size
                      << std::right << std::setw(20) << avg_ns
                      << std::right << std::setw(18) << ms_s.str() << '\n';
        }
    }
};

int main(int argc, char** argv) {
    int iters = 10;
    std::vector<int> sizes = {64, 128, 192, 256, 384, 512};
    bool csv = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench_basic [--iters=N] [--sizes=LIST|--sizes LIST] [--csv]" << '\n';
            return 0;
        } else if (arg.rfind("--iters=", 0) == 0) {
            iters = std::stoi(arg.substr(8));
        } else if (arg == "--sizes" && i + 1 < argc) {
            sizes.clear();
            std::string list = argv[++i];
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                sizes.push_back(std::stoi(list.substr(pos, comma - pos)));
                pos = comma + (comma < list.size());
            }
        } else if (arg.rfind("--sizes=", 0) == 0) {
            sizes.clear();
            std::string list = arg.substr(8);
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                sizes.push_back(std::stoi(list.substr(pos, comma - pos)));
                pos = comma + (comma < list.size());
            }
        } else if (arg == "--csv") {
            csv = true;
        }
    }

    BenchPrinter printer(csv);
    printer.print_header();

    for (int n : sizes) {
        // Matmul timing: C = A * B
        {
            myla::Matrix<double, myla::Dynamic, myla::Dynamic> A(n, n);
            myla::Matrix<double, myla::Dynamic, myla::Dynamic> B(n, n);
            myla::Matrix<double, myla::Dynamic, myla::Dynamic> C(n, n);
            fill_random(A.data(), A.size(), 1u);
            fill_random(B.data(), B.size(), 2u);
            auto avg_ns = time_function_ns([&]{
                // naive matmul provided by library
                C = A.matmul(B);
            }, iters);
            printer.print_row("matmul_myla", n, avg_ns);

            // Eigen comparison
            Eigen::MatrixXd AE(n, n), BE(n, n), CE(n, n);
            fill_random(AE.data(), static_cast<size_t>(AE.size()), 1u);
            fill_random(BE.data(), static_cast<size_t>(BE.size()), 2u);
            auto avg_ns_e = time_function_ns([&]{
                CE.noalias() = AE * BE;
            }, iters);
            printer.print_row("matmul_eigen", n, avg_ns_e);
        }

        // LU + determinant
        {
            myla::Matrix<double, myla::Dynamic, myla::Dynamic> A(n, n);
            fill_random(A.data(), A.size(), 3u);
            // Make it diagonally dominant to avoid singularities
            for (int i = 0; i < n; ++i) A(i,i) += n;
            volatile double sink = 0.0;
            auto avg_ns = time_function_ns([&]{
                auto lu = myla::lu_decompose(A);
                sink += static_cast<double>(myla::determinant(lu));
            }, iters);
            printer.print_row("lu_det_myla", n, avg_ns);
            (void)sink;
            // Eigen comparison
            Eigen::MatrixXd AE(n, n);
            fill_random(AE.data(), static_cast<size_t>(AE.size()), 3u);
            for (int i = 0; i < n; ++i) AE(i,i) += n;
            volatile double sinkE = 0.0;
            auto avg_ns_e = time_function_ns([&]{
                Eigen::PartialPivLU<Eigen::MatrixXd> lu(AE);
                sinkE += lu.determinant();
            }, iters);
            printer.print_row("lu_det_eigen", n, avg_ns_e);
            (void)sinkE;
        }

        // LU solve
        {
            myla::Matrix<double, myla::Dynamic, myla::Dynamic> A(n, n);
            myla::Matrix<double, myla::Dynamic, 1> b(n, 1);
            fill_random(A.data(), A.size(), 4u);
            for (int i = 0; i < n; ++i) A(i,i) += n;
            fill_random(b.data(), b.size(), 5u);
            volatile double sink = 0.0;
            auto avg_ns = time_function_ns([&]{
                auto lu = myla::lu_decompose(A);
                auto x = myla::lu_solve(lu, b);
                sink += x(0,0);
            }, iters);
            printer.print_row("lu_solve_myla", n, avg_ns);
            (void)sink;

            // Eigen comparison
            Eigen::MatrixXd AE(n, n);
            Eigen::VectorXd bE(n);
            fill_random(AE.data(), static_cast<size_t>(AE.size()), 4u);
            for (int i = 0; i < n; ++i) AE(i,i) += n;
            fill_random(bE.data(), static_cast<size_t>(bE.size()), 5u);
            volatile double sinkE = 0.0;
            auto avg_ns_e = time_function_ns([&]{
                Eigen::PartialPivLU<Eigen::MatrixXd> lu(AE);
                Eigen::VectorXd x = lu.solve(bE);
                sinkE += x(0);
            }, iters);
            printer.print_row("lu_solve_eigen", n, avg_ns_e);
            (void)sinkE;
        }

        // Inverse
        {
            myla::Matrix<double, myla::Dynamic, myla::Dynamic> A(n, n);
            fill_random(A.data(), A.size(), 6u);
            for (int i = 0; i < n; ++i) A(i,i) += n;
            volatile double sink = 0.0;
            auto avg_ns = time_function_ns([&]{
                auto invA = A.inverse();
                sink += invA(0,0);
            }, iters);
            printer.print_row("inverse_myla", n, avg_ns);
            (void)sink;

            // Eigen comparison
            Eigen::MatrixXd AE(n, n);
            fill_random(AE.data(), static_cast<size_t>(AE.size()), 6u);
            for (int i = 0; i < n; ++i) AE(i,i) += n;
            volatile double sinkE = 0.0;
            auto avg_ns_e = time_function_ns([&]{
                Eigen::MatrixXd invA = AE.inverse();
                sinkE += invA(0,0);
            }, iters);
            printer.print_row("inverse_eigen", n, avg_ns_e);
            (void)sinkE;
        }
    }

    return 0;
}


