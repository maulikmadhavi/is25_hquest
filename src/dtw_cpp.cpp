#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace py = pybind11;

// Optimized DTW implementation
class DTWComputer {
private:
    std::vector<std::vector<double>> cost_mat;
    std::vector<std::vector<double>> dist_mat;
    
public:
    // Euclidean distance computation
    static double euclidean_distance(double a, double b) {
        double diff = a - b;
        return std::sqrt(diff * diff);  // Return actual Euclidean distance
    }
    
    // Fast DTW using dynamic programming with space optimization
    double compute_dtw_distance(py::array_t<double> query, py::array_t<double> audio) {
        auto query_buf = query.request();
        auto audio_buf = audio.request();
        
        if (query_buf.ndim != 1 || audio_buf.ndim != 1) {
            throw std::runtime_error("Input arrays must be 1D");
        }
        
        double* query_ptr = static_cast<double*>(query_buf.ptr);
        double* audio_ptr = static_cast<double*>(audio_buf.ptr);
        
        int N = query_buf.shape[0];
        int M = audio_buf.shape[0];
        
        if (N == 0 || M == 0) {
            return std::numeric_limits<double>::infinity();
        }
        
        // Resize cost matrix
        cost_mat.assign(N + 1, std::vector<double>(M + 1, std::numeric_limits<double>::infinity()));
        cost_mat[0][0] = 0.0;
        
        // Fill DP table
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= M; ++j) {
                double dist = euclidean_distance(query_ptr[i-1], audio_ptr[j-1]);
                double min_prev = std::min({cost_mat[i-1][j], 
                                           cost_mat[i][j-1], 
                                           cost_mat[i-1][j-1]});
                cost_mat[i][j] = dist + min_prev;
            }
        }
        
        return cost_mat[N][M];
    }
    
    // Batch DTW computation for multiple audio sequences
    py::array_t<double> compute_dtw_batch(py::array_t<double> query, 
                                          py::list audio_sequences) {
        auto query_buf = query.request();
        double* query_ptr = static_cast<double*>(query_buf.ptr);
        int N = query_buf.shape[0];
        
        int num_sequences = audio_sequences.size();
        std::vector<double> distances(num_sequences);
        
        for (int idx = 0; idx < num_sequences; ++idx) {
            py::array_t<double> audio = py::cast<py::array_t<double>>(audio_sequences[idx]);
            auto audio_buf = audio.request();
            double* audio_ptr = static_cast<double*>(audio_buf.ptr);
            int M = audio_buf.shape[0];
            
            if (M == 0) {
                distances[idx] = std::numeric_limits<double>::infinity();
                continue;
            }
            
            // Compute DTW for this sequence
            cost_mat.assign(N + 1, std::vector<double>(M + 1, std::numeric_limits<double>::infinity()));
            cost_mat[0][0] = 0.0;
            
            for (int i = 1; i <= N; ++i) {
                for (int j = 1; j <= M; ++j) {
                    double dist = euclidean_distance(query_ptr[i-1], audio_ptr[j-1]);
                    double min_prev = std::min({cost_mat[i-1][j], 
                                               cost_mat[i][j-1], 
                                               cost_mat[i-1][j-1]});
                    cost_mat[i][j] = dist + min_prev;
                }
            }
            
            distances[idx] = cost_mat[N][M];
        }
        
        return py::array_t<double>(distances);
    }
};

PYBIND11_MODULE(dtw_cpp, m) {
    m.doc() = "Fast DTW implementation in C++";
    
    py::class_<DTWComputer>(m, "DTWComputer")
        .def(py::init<>())
        .def("compute_dtw_distance", &DTWComputer::compute_dtw_distance,
             "Compute DTW distance between two sequences")
        .def("compute_dtw_batch", &DTWComputer::compute_dtw_batch,
             "Compute DTW distances for multiple sequences");
}
