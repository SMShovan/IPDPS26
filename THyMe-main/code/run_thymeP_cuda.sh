nvcc -O2 -std=c++17 main_thymeP_cuda.cu -o run_thymeP_cuda

dataset=email-Enron-full
delta=86400000
./run_thymeP_cuda $dataset $delta 8
rm run_thymeP_cuda
