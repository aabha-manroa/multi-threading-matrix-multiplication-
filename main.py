import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import psutil

# Step 1: Generate the matrices
print("Generating matrices...")
num_matrices = 500
matrix_size = 5000
matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(num_matrices)]
constant_matrix = np.random.rand(matrix_size, matrix_size)
print("Matrices generated.")

# Step 2: Define the matrix multiplication function
def multiply_matrices(matrix):
    return np.dot(matrix, constant_matrix)

# Step 3: Execute multithreaded function
def execute_multithreaded(num_threads):
    start_time = time.time()
    print(f"Running with {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(multiply_matrices, matrices))
    end_time = time.time()
    print(f"Execution completed with {num_threads} threads.")
    return (end_time - start_time) / 60  # Return time in minutes

# Step 4: Run for different thread counts and record time
times = {}
for threads in range(1, 9):  # From T=1 to T=8
    exec_time = execute_multithreaded(threads)
    times[threads] = exec_time
    print(f"Threads={threads}, Time Taken={exec_time:.2f} mins")

# Step 5: Plot the execution time against number of threads
print("Plotting the graph...")
plt.plot(list(times.keys()), list(times.values()), marker='o', color='b')
plt.xlabel('Number of Threads')
plt.ylabel('Time Taken (mins)')
plt.title('Execution Time vs Number of Threads')
plt.grid(True)
plt.show()

# Step 6: Display CPU usage
print("Displaying CPU usage...")
cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
print("CPU usage per core:", cpu_usage)
