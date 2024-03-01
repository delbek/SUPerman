#include "argparse.hpp"

struct Args : argparse::Args {
    std::string &filename = arg("filename", "The path to the matrix file");
    int &algo_id = kwarg("a,algo", "The algorithm to use").set_default(1);
    int &threads = kwarg("t,threads", "The number of threads to use").set_default(16); 
    int &devices = kwarg("d,devices", "The number of devices to be used in a multi-GPU algorithm").set_default(2);
    int &trials = kwarg("x,trials", "The number of trials to run for an approximation algorithm").set_default(100000);

    bool &sparse = flag("s,sparse", "Runs the sparse algorithm");
    bool &approximation = flag("a,approximation", "Runs the approximation algorithm");
    bool &binary = flag("b,binary", "Treats the matrix as a binary matrix where all nonzero values are taken to be 1");
    bool &gpu = flag("g,gpu", "Runs the algorithm on the GPU");
    bool &cpu = flag("c,cpu", "Runs the algorithm on the CPU (if given with -g, runs a hybrid algorithm)");
    int &preprocessing = kwarg("r,preprocessing", "Preprocessing to be applied (1: SortOrder, 2: SkipOrder)").set_default(0); 

    bool &grid = flag("i,grid", "Creates a grid graph and uses a sparse approximation algorithm"); 
    int &gridm = kwarg("m,grid-m", "First dimension of the grid graph").set_default(36); 
    int &gridn = kwarg("n,grid-n", "Second dimension of the grid graph").set_default(36);

    bool &use32bitCalculation = flag("calculate-32bit", "Use 32 bit data type for calculation");
    bool &use128bitCalculation = flag("calculate-128bit", "Use 128 bit data type for calculation (CPU Only)");
    bool &use32bitStorage = flag("storage-32bit", "Use 32 bit data type for storage");  
    bool &use128bitStorage = flag("storage-128bit", "Use 128 bit data type for storage (CPU Only)");

    int &gpu_id = kwarg("gpu-id", "GPU id to run single GPU algorithm").set_default(0); 
    int &repetitions = kwarg("k", "The number of times to run the algorithm independently").set_default(1); 
    int &cudaGridMultiplier = kwarg("e", "CUDA grid dimension multiplier").set_default(1);
    bool &compression = flag("o", "Enable compression"); 

    double &scaleValue = kwarg("u", "Scale input matrix to value").set_default(1.0); 
    int &scaleIntervals = kwarg("y,scale-intervals", "Scale intervals for a scaling approximation algorithm").set_default(4);
    int &scaleTimes = kwarg("z,scale-times", "Number of times to scale for a scaling approximation algorithm").set_default(5); 
};
