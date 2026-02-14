/**
 * Parallel Solver Implementation using C++11 std::thread
 * 
 * Alternative to OpenMP for MinGW 6.3.0 which lacks pthread.
 * Uses standard C++ threading for parallelization.
 */

#ifndef PARALLEL_SOLVER_H
#define PARALLEL_SOLVER_H

#include <vector>
#include <thread>
#include <functional>
#include <atomic>

/**
 * Simple parallel for-loop implementation using std::thread
 * 
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to execute for each index
 * @param num_threads Number of threads (default: hardware concurrency)
 */
template<typename Func>
void parallel_for(int start, int end, Func func, int num_threads = 0) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;  // fallback
    }
    
    int total = end - start;
    if (total <= 0) return;
    
    // If work is small, don't bother with threads
    if (total < num_threads * 10) {
        for (int i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    int chunk_size = (total + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        int chunk_start = start + t * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, end);
        
        if (chunk_start >= end) break;
        
        threads.emplace_back([chunk_start, chunk_end, &func]() {
            for (int i = chunk_start; i < chunk_end; ++i) {
                func(i);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

/**
 * Parallel for with 2D indexing (for collapse-like behavior)
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param func Function taking (row, col) indices
 * @param num_threads Number of threads
 */
template<typename Func>
void parallel_for_2d(int rows, int cols, Func func, int num_threads = 0) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }
    
    int total = rows * cols;
    if (total <= 0) return;
    
    // For small grids, run sequentially
    if (total < 100) {
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                func(j, i);
            }
        }
        return;
    }
    
    // Parallelize over rows
    int rows_per_thread = (rows + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = std::min(row_start + rows_per_thread, rows);
        
        if (row_start >= rows) break;
        
        threads.emplace_back([row_start, row_end, cols, &func]() {
            for (int j = row_start; j < row_end; ++j) {
                for (int i = 0; i < cols; ++i) {
                    func(j, i);
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

/**
 * Parallel reduction (sum)
 */
template<typename T>
T parallel_reduce(int start, int end, std::function<T(int)> func, int num_threads = 0) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }
    
    int total = end - start;
    if (total <= 0) return T{};
    
    int chunk_size = (total + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    std::vector<T> partial_sums(num_threads, T{});
    
    for (int t = 0; t < num_threads; ++t) {
        int chunk_start = start + t * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, end);
        
        if (chunk_start >= end) break;
        
        threads.emplace_back([chunk_start, chunk_end, &func, &partial_sums, t]() {
            T local_sum{};
            for (int i = chunk_start; i < chunk_end; ++i) {
                local_sum += func(i);
            }
            partial_sums[t] = local_sum;
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    T total_sum{};
    for (const auto& ps : partial_sums) {
        total_sum += ps;
    }
    return total_sum;
}

#endif // PARALLEL_SOLVER_H
