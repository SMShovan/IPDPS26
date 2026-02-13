g++ -O3 -std=c++11 -fopenmp main_thymeP.cpp -o run_thymeP;

dataset=email-Enron-full
delta=86400000
./run_thymeP $dataset $delta;
rm run_thymeP;