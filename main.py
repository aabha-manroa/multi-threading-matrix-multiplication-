import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
import psutil
import tracemalloc

# Define the matrix multiplication function outside of main
def multiply_matrices(matrix_constant_tuple):
    matrix, constant_matrix = matrix_constant_tuple
    return np.dot(matrix, constant_matrix)

def execute_multiprocessing(num_threads, matrices, constant_matrix):
    start_time = time.time()
    print(f"Running with {num_threads} processes...")
    
    # Prepare the data as tuples of (matrix, constant_matrix) for each multiplication
    data = [(matrix, constant_matrix) for matrix in matrices]
    
    with Pool(processes=num_threads) as pool:
        results = pool.map(multiply_matrices, data, chunksize=1)
    
    end_time = time.time()
    print(f"Execution completed with {num_threads} processes.")
    return (end_time - start_time) / 60  # Return time in minutes

def main():
    # Adjusted parameters to test in a manageable way
    num_matrices = 5    # Start with 5 matrices
    matrix_size = 10    # Smaller matrix size for testing

    # Generate smaller matrices
    print("Generating matrices...")
    matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(num_matrices)]
    constant_matrix = np.random.rand(matrix_size, matrix_size)
    print("Matrices generated.")

    # Run for a single thread to test
    try:
        tracemalloc.start()  # Start memory monitoring
        exec_time = execute_multiprocessing(1, matrices, constant_matrix)  # Start with 1 thread only for testing
        print(f"Time Taken={exec_time:.2f} mins")

        # Plot the execution time (for demonstration)
        plt.plot([1], [exec_time], marker='o', color='b')
        plt.xlabel('Number of Processes')
        plt.ylabel('Time Taken (mins)')
        plt.title('Execution Time with Limited Process')
        plt.grid(True)
        plt.show()
        
        # Display CPU and memory usage
        print("Displaying CPU usage and memory...")
        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
        print("CPU usage per core:", cpu_usage)
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10**6:.2f} MB; Peak: {peak / 10**6:.2f} MB")
        tracemalloc.stop()
        
    except Exception as e:
        print("Execution stopped due to an error:", e)

# Ensuring code runs in the main block
if __name__ == '__main__':
    main()
