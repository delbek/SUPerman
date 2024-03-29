#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <omp.h>
#include <stdio.h>
#include "util.h" //evaluate_data_return_parameters() --> To be implemented
#include <iomanip> //To debug avx vectors clearly

#ifndef ONLYCPU
#include "gpu_wrappers.h" //All GPU wrappers will be stored there to simplify things
#else
#include "flags.h"

#ifdef AVX
#include <immintrin.h>
#endif

#endif

//
#include "cpu_algos.hpp"
//
#include <math.h>
//
#include "read_matrix.hpp"
//
#include "MatrixMarketIOLibrary.h"

#include <cfenv>

#include <unistd.h>
#include <limits.h>
//#define STRUCTURAL
//#define HEAVYDEBUG

#include "args.hpp"

using namespace std;

int recursive_count = 0;
int RANK;
int NPROCS;

template<class S>
bool max20(DenseMatrix<S>* densemat){

  int nov = densemat->nov;
  

  for(int i = 0; i < nov*nov; i++){
    if(densemat->mat[i] > (S)20)
      return true;
    }

  return false;
}

template<class S>
Result scale_and_calculate(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags, bool compressing);

void print_flags(flags flags){
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  //
  if(RANK == 0) std::cout << "*~~~~~~~~~~~~FLAGS~~~~~~~~~~~~*" << std::endl;
  if(RANK == 0) std::cout << "- cpu: " << flags.cpu << std::endl;
  if(RANK == 0) std::cout << "- gpu: " << flags.gpu << std::endl;
  if(RANK == 0) std::cout << "- sparse: " << flags.sparse << std::endl;
  if(RANK == 0) std::cout << "- dense: " << flags.dense << std::endl;
  if(RANK == 0) std::cout << "- exact: " << flags.exact << std::endl;
  if(RANK == 0) std::cout << "- approximation: " << flags.approximation << std::endl;
  if(RANK == 0) std::cout << "- calculation half-precision: " << flags.calculation_half_precision << std::endl;
  if(RANK == 0) std::cout << "- calculation quad-precision: " << flags.calculation_quad_precision << std::endl;
  if(RANK == 0) std::cout << "- storage half-precision: " << flags.storage_half_precision << std::endl;
  if(RANK == 0) std::cout << "- storage quad-precision: " << flags.storage_quad_precision << std::endl;
  if(RANK == 0) std::cout << "- binary graph: " << flags.binary_graph << std::endl;
  if(RANK == 0) std::cout << "- grid_graph: " << flags.grid_graph << std::endl;
  if(RANK == 0) std::cout << "- gridm: " << flags.gridm << std::endl;
  if(RANK == 0) std::cout << "- gridn: " << flags.gridn << std::endl;
  if(RANK == 0) std::cout << "- perman_algo: " << flags.perman_algo << std::endl;
  if(RANK == 0) std::cout << "- threads: " << flags.threads << std::endl;
  if(RANK == 0) std::cout << "- scale_intervals: " << flags.scale_intervals << std::endl;
  if(RANK == 0) std::cout << "- scale_times: " << flags.scale_times << std::endl;
  if(RANK == 0) printf("- fname: %s \n", flags.filename);
  if(RANK == 0) std::cout << "- type: " << flags.type << std::endl;
  if(RANK == 0) std::cout << "- no rep.: " << flags.rep << std::endl;
  if(RANK == 0) std::cout << "- preprocessing: " << flags.preprocessing << std::endl;
  if(RANK == 0) std::cout << "- gpu_num: " << flags.gpu_num << std::endl;
  if(RANK == 0) std::cout << "- number_of_times: " << flags.number_of_times << std::endl;
  if(RANK == 0) std::cout << "- grid_dim: " << flags.grid_dim << std::endl;
  if(RANK == 0) std::cout << "- block_dim: " << flags.block_dim << std::endl;
  if(RANK == 0) std::cout << "- device_id: " << flags.device_id << std::endl;
  if(RANK == 0) std::cout << "- grid_multip: " << flags.grid_multip << std::endl;
  if(RANK == 0) std::cout << "- compression: " << flags.compression << std::endl;
  if(RANK == 0) std::cout << "- scaling_threshold: " << flags.scaling_threshold << std::endl;
  if(RANK == 0) std::cout << "- sync_gray: " << flags.synchronized_gray << std::endl;
  if(RANK == 0) std::cout << "- hostname: " << hostname << std::endl;
  if(RANK == 0) std::cout << "*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;
  //
}

template <class T>
DenseMatrix<double>* avx_help(DenseMatrix<T>* densemat){

  int nov = densemat->nov;
  
  DenseMatrix<double>* ret_mat = new DenseMatrix<double>;

  ret_mat->nov = densemat->nov;
  ret_mat->nnz = densemat->nnz;

  ret_mat->mat = new double[nov*nov];

  for(int i = 0; i < nov*nov; i++){
    ret_mat->mat[i] = (double)densemat->mat[i];
  }

  return ret_mat;
  
}

template <class S>
Result RunAlgo(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags, bool supress) 
{

  //int grid_dim = 2048; //For any case, if it's failed to determined by CUDA
  //int block_dim = 256;
  
  //Pack flags
  bool cpu = flags.cpu;
  bool gpu = flags.gpu;

  bool dense = flags.dense;
  bool sparse = flags.sparse;

  bool exact = flags.exact;
  bool approximation = flags.approximation;
  
  int perman_algo = flags.perman_algo;
  int gpu_num = flags.gpu_num;
  int threads = flags.threads;

  int number_of_times = flags.number_of_times;
  int scale_intervals = flags.scale_intervals;
  int scale_times = flags.scale_times;
  int no_repetition = flags.rep;
  //Pack flags
  
  double start, end, perman;
  
  Result result;
  
  if(cpu && dense && exact && !gpu){    
    
    if(perman_algo == 1){
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_perman";
      if(flags.calculation_half_precision)
	result = parallel_perman64<float, S>(densemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_perman64<__float128, S>(densemat, flags);
      else
	result = parallel_perman64<double, S>(densemat, flags);	
    }


#ifdef AVX
    else if(perman_algo == 2){
#ifdef STRUCTURAL
      exit(1);
#endif
      
      DenseMatrix<double>* avx_helped = avx_help(densemat);
      
      flags.algo_name = "parallel_perman_avx512";
      result = parallel_perman64_avx512<double, double>(avx_helped, flags);
      
      delete avx_helped;
    }

    else if(perman_algo == 3){
#ifdef STRUCTURAL
      exit(1);
#endif
      
      DenseMatrix<double>* avx_helped = avx_help(densemat);
      
      flags.algo_name = "parallel_perman_avx512_32bit";
      result = parallel_perman64_avx512_32bit<double, double>(avx_helped, flags);
      
      delete avx_helped;
    }
#endif
    
    
    else{
      if(RANK == 0) std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
    
  }
  
  if(cpu && sparse && exact && !gpu){
    
    if (perman_algo == 1) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_perman_sparse";
      if(flags.calculation_half_precision)
	result = parallel_perman64_sparse<float, S>(densemat, sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_perman64_sparse<__float128, S>(densemat, sparsemat, flags);
      else
	result = parallel_perman64_sparse<double, S>(densemat, sparsemat, flags); 
    }
    else if (perman_algo == 2) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_skip_perman";

      if(flags.calculation_half_precision)
	result = parallel_skip_perman64_w<float, S>(sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_skip_perman64_w<__float128, S>(sparsemat, flags);
      else
	result = parallel_skip_perman64_w<double, S>(sparsemat, flags);
    }
    else if (perman_algo == 3) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_skip_perman_balanced";
      if(flags.calculation_half_precision)
	result = parallel_skip_perman64_w_balanced<float, S>(sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_skip_perman64_w_balanced<__float128, S>(sparsemat, flags);
      else
	result = parallel_skip_perman64_w_balanced<double, S>(sparsemat, flags);
    }
    else
      {
	if(RANK == 0) std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
	exit(1);
      }
  }
  
#ifndef ONLYCPU
  if(gpu && dense && exact && !cpu){
    
    if (perman_algo == 21) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xglobal";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xglobal<float, S>(densemat, flags);
      else
	result = gpu_perman64_xglobal<double, S>(densemat, flags);
    }
    else if (perman_algo == 1) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xlocal";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xlocal<float, S>(densemat, flags);
      else
	result = gpu_perman64_xlocal<double, S>(densemat, flags);
      
    }
    else if (perman_algo == 2) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared<double, S>(densemat, flags);
      
    }

    ////////
    
    else if (perman_algo == 3) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing<double, S>(densemat, flags);
    }

    else if (perman_algo == 31) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_plainmatrix";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_plainmatrix<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_plainmatrix<double, S>(densemat, flags);
    }

    else if (perman_algo == 32) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_plainmatrix_texfour";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_plainmatrix_texfour<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_plainmatrix_texfour<double, S>(densemat, flags);
    }

    else if (perman_algo == 33) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_plainmatrix_texeight";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_plainmatrix_texeight<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_plainmatrix_texeight<double, S>(densemat, flags);
    }

    else if (perman_algo == 34) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_plainmatrix_mshared";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_plainmatrix_mshared<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_plainmatrix_mshared<double, S>(densemat, flags);
    }

    else if (perman_algo == 35) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xregister_coalescing_plainmatrix_mshared";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared<float, S>(densemat, flags);
      else
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared<double, S>(densemat, flags);
    }

    else if (perman_algo == 311) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xregister_coalescing_plainmatrix_mshared_cgray";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared_cgray<float, S>(densemat, flags);
      else
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared_cgray<double, S>(densemat, flags);
    }

    else if (perman_algo == 312) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xregister_coalescing_cgray";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xregister_coalescing_cgray<float, S>(densemat, flags);
      else
	result = gpu_perman64_xregister_coalescing_cgray<double, S>(densemat, flags);
    }
    
    else if (perman_algo == 36) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_register_coalescing_mshared_mpi";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared_mpi<float, S>(densemat, flags);
      else
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared_mpi<double, S>(densemat, flags);
    }

    else if (perman_algo == 37) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_plainmatrix_mshared_selected_perwarp";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_plainmatrix_mshared_selected_perwarp<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_plainmatrix_mshared_selected_perwarp<double, S>(densemat, flags);
    }

    else if (perman_algo == 38) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xregister_coalescing_plainmatrix_mshared_selected_perwarp";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared_selected_perwarp<float, S>(densemat, flags);
      else
	result = gpu_perman64_xregister_coalescing_plainmatrix_mshared_selected_perwarp<double, S>(densemat, flags);
    }
    /////
    
    else if (perman_algo == 4) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared<double, S>(densemat, flags);
      
    }
    //COALESCED GRAY
    else if (perman_algo == 41) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_cgray";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_cgray<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_cgray<double, S>(densemat, flags);
      
    }
    //COALESCED GRAY
    else if (perman_algo == 5) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman64_xshared_mshared_multigpu";

      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpu<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpu<double, S>(densemat, flags);
    }
    else if (perman_algo == 6) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpu_manual_distribution";
      flags.gpu_num = 4; //This will change accordingly
      
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(densemat, flags);	
    }
    else if (perman_algo == 7) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks<double, S>(densemat, flags);
    }
    
    else{
      if(RANK == 0) std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  if(gpu && sparse && exact && !cpu){
    
    if (perman_algo == 1) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xlocal_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xlocal_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xlocal_sparse<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 2) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_sparse<double, S>(densemat, sparsemat, flags);
      
    }
    else if (perman_algo == 3) {
#ifdef STRUCTURAL
      exit(1);
#endif
      
      flags.algo_name = "gpu_perman_xshared_coalescing_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_sparse<double, S>(densemat, sparsemat, flags);
      
      
    }
    else if (perman_algo == 4) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_sparse<double, S>(densemat, sparsemat, flags);
      
    }
    else if (perman_algo == 5) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpu_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 7) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 14){
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_skipper";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_skipper<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_skipper<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 17){
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_skipper";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 6) {
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpu_sparse_manual_distribution";
      flags.gpu_num = 4; //This is a manual setting specialized for GPUs we have, so recommend not to use it.
	perman = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution(densemat, sparsemat, flags);

    
    }
    else{
    if(RANK == 0) std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  if(gpu && cpu && dense && exact){
    if (perman_algo == 7) {
#ifdef STRUCTURAL
      exit(1);
#endif
      
      flags.algo_name = "gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks<double, S>(densemat, flags);   
    }
  }
  
  if (gpu && cpu && sparse && exact) {
    
    if(perman_algo == 7){
      
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse<double, S>(densemat, sparsemat, flags);
    }
    
    else if(perman_algo == 17){
#ifdef STRUCTURAL
      exit(1);
#endif
      
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_skipper";
      //This will change accordingly
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper<double, S>(densemat, sparsemat, flags);
    }
  }
  
#endif
  return result;
}

template<class S>
int getNnz(S* mat2, int nov2){
  int nnz2 = 0;

  for(int i = 0; i < nov2*nov2; i++){
    if(mat2[i] > (S)0)
      nnz2++;
  }

  return nnz2;
}

template<class S>
DenseMatrix<S>* create_densematrix_from_mat2(S* mat2, int nov2){
  
  DenseMatrix<S>* densemat2 = new DenseMatrix<S>();
  densemat2->mat = mat2;
  densemat2->nov = nov2;
  densemat2->nnz = getNnz(mat2, nov2);
  
  return densemat2;
}

template<class S>
SparseMatrix<S>* create_sparsematrix_from_densemat2(DenseMatrix<S>* densemat2, flags flags){

  int nnz = densemat2->nnz;
  int nov = densemat2->nov;
  
  SparseMatrix<S>* sparsemat2 = new SparseMatrix<S>();
  sparsemat2->rvals = new S[nnz];
  sparsemat2->cvals = new S[nnz];
  sparsemat2->cptrs = new int[nov + 1];
  sparsemat2->rptrs = new int[nov + 1];
  sparsemat2->rows = new int[nnz];
  sparsemat2->cols = new int[nnz];
  sparsemat2->nov = nov;
  sparsemat2->nnz = nnz;

  if(flags.preprocessing == 0)
    matrix2compressed_o(densemat2, sparsemat2);
  if(flags.preprocessing == 1)
    matrix2compressed_sortOrder_o(densemat2, sparsemat2);
  if(flags.preprocessing == 2)
    matrix2compressed_skipOrder_o(densemat2, sparsemat2);

  return sparsemat2;
}

template<class S>
Result compress_and_calculate_recursive(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags){


  
  bool silent = 0;
  
  Result result;
  int minDeg = getMinNnz(densemat->mat, densemat->nov);
  cout << "##MINDEG: " << minDeg << endl;
  if(minDeg < 5 && densemat->nov > 30){

    if(minDeg == 1){
      d1compress(densemat->mat, densemat->nov);
      if(!silent)
	cout << "d1: matrix is reduced to: " << densemat->nov << " rows" << endl;
      densemat->nnz = getNnz(densemat->mat, densemat->nov);
      delete sparsemat;
      sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
      return compress_and_calculate_recursive(densemat, sparsemat, flags);
    }
    
    else if(minDeg == 2){
      d2compress(densemat->mat, densemat->nov);
      if(!silent)
	cout << "d2: matrix is reduced to: " << densemat->nov << " rows" << endl;
      densemat->nnz = getNnz(densemat->mat, densemat->nov);
      delete sparsemat;
      sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
      return compress_and_calculate_recursive(densemat, sparsemat, flags);
    }
    
    else if(minDeg == 3 || minDeg == 4){
      S* mat2 = nullptr;
      int nov2;
      d34compress(densemat->mat, densemat->nov, mat2, nov2, minDeg);
      DenseMatrix<S>* densemat2 = create_densematrix_from_mat2(mat2, nov2);
      //Don't forget realign compressed matrices features
      densemat->nnz = getNnz(densemat->mat, densemat->nov);
      sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
      densemat2->nnz = getNnz(densemat2->mat, densemat2->nov);
      SparseMatrix<S>* sparsemat2 = create_sparsematrix_from_densemat2(densemat2, flags);
      
      if(!silent)
	cout << "d34: matrix is reduced to matrices with: " << densemat->nov << " and " << densemat2->nov <<" rows" << endl;
      
      result = compress_and_calculate_recursive(densemat, sparsemat, flags) +
	compress_and_calculate_recursive(densemat2, sparsemat2, flags);

      if(mat2 != nullptr){
	delete[] mat2;
	mat2 = nullptr;
      }
    }
  }
  else{

#ifndef ONLYCPU
    if(flags.scaling_threshold != -1.0)
      result = scale_and_calculate(densemat, sparsemat, flags, true);
    else
      result = RunAlgo(densemat, sparsemat, flags, true);
#else
    result = RunAlgo(densemat, sparsemat, flags, true);
#endif

  }
  return result;
}

template<class S>
Result compress_singleton_and_then_recurse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags){

  bool comp = true;
  while(comp && densemat->nov > 1){
    comp = d1compress(densemat->mat, densemat->nov);
    if(comp){
      cout << "Removing singleton -- d1: matrix is reduced to: " << densemat->nov << " rows" << endl;
      densemat->nnz = getNnz(densemat->mat, densemat->nov);
      //delete sparsemat; //Possible memory leak, will deal with later
      sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
    }
    else{
      comp = d2compress(densemat->mat, densemat->nov);
      if(comp){
	cout << "Removing singleton -- d2: matrix is reduced to: " << densemat->nov << " rows" << endl;
	densemat->nnz = getNnz(densemat->mat, densemat->nov);
	//delete sparsemat; //Possible memory leak, will deal with later
	sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
      }
    }

    if(comp){
      if(checkEmpty(densemat->mat, densemat->nov)){
	cout << "Matrix is rank deficient!" << endl;
	cout << "Perman is 0" << endl;
	exit(1);
      }
    }
  }
  
  cout << "Singleton compressing is done" << endl;
  return compress_and_calculate_recursive(densemat, sparsemat, flags);
  
}

template<class S>
Result scale_and_calculate(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags, bool compressing){

  //Do not delete sparsemat inside of scale_and_calculate because it is already deleted in main
  
    
  //Pack parameters//
  int nov = densemat->nov;
  //Pack parameters//

  Result result;
  
  if(!flags.storage_half_precision && flags.type == "int"){
    
    flags.type = "double";
    
    DenseMatrix<double>* densemat2 = swap_types<S, double>(densemat);
    
    //delete sparsemat;
    
    SparseMatrix<double>* sparsemat2 = create_sparsematrix_from_densemat2(densemat2, flags);
    ScaleCompanion<double>* sc = scalesk(sparsemat2, flags);
    scaleMatrix(densemat2, sc);
    delete sparsemat2;
    
    SparseMatrix<double>* sparsemat3 = create_sparsematrix_from_densemat2(densemat2, flags);
    
    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat2, sparsemat3, flags);
    else
      result = RunAlgo(densemat2, sparsemat3, flags, false);
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }
    
  }
  
  
  else if(flags.storage_half_precision && flags.type == "int"){
    
    flags.type = "float";
  
    DenseMatrix<float>* densemat2 = swap_types<S, float>(densemat);

    
    //delete densemat;
    delete sparsemat;
    
    SparseMatrix<float>* sparsemat2 = create_sparsematrix_from_densemat2(densemat2, flags);
    ScaleCompanion<float>* sc = scalesk(sparsemat2, flags);
    scaleMatrix(densemat2, sc);
    delete sparsemat2;
    
    SparseMatrix<float>* sparsemat3 = create_sparsematrix_from_densemat2(densemat2, flags);

    
    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat2, sparsemat3, flags);
    else
      result = RunAlgo(densemat2, sparsemat3, flags, false);

    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }
    
  }
  
  else if(!flags.storage_half_precision && flags.type == "double"){
    ScaleCompanion<S>* sc = scalesk(sparsemat, flags);
    scaleMatrix(densemat, sc);
    //delete sparsemat;
    
    SparseMatrix<S>* sparsemat2 = create_sparsematrix_from_densemat2(densemat, flags);

    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat, sparsemat2, flags);
    else
      result = RunAlgo(densemat, sparsemat2, flags, false);


    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }
  }
  
  
  else if(flags.storage_half_precision && flags.type == "float"){
    ScaleCompanion<S>* sc = scalesk(sparsemat, flags);
    scaleMatrix(densemat, sc);
    //delete sparsemat;
    
    SparseMatrix<S>* sparsemat2 = create_sparsematrix_from_densemat2(densemat, flags);
        
    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat, sparsemat2, flags);
    else
      result = RunAlgo(densemat, sparsemat2, flags, false);
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }  
  }

  else{
    if(RANK == 0) std::cout << "Why do you want to scale? Exiting.. " << std::endl;
    exit(1);
  }

  return result;
}




int main (int argc, char **argv)
{

#ifdef MPIENABLED
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
  MPI_Comm_size(MPI_COMM_WORLD, &NPROCS);

  if(RANK == 0) if(RANK==0) printf("==SC== MPI Initialized.. \n");
#else
  RANK = 0;
#endif
  
  if(RANK == 0) {
    std::cout << "**command: ";
    for(int i = 0 ; i < argc; i++){
      std::cout << argv[i] << " ";
    }
    std::cout << std::endl;
  }
  

  auto args = argparse::parse<Args>(argc, argv);
  flags flags = args.to_flags();
  
  if (!flags.grid_graph && flags.filename == "") {
    fprintf (stderr, "Filename required as positional argument. See help with --help.\n");
    return 1;
  }
  
  if (!flags.cpu && !flags.gpu) {
    flags.gpu = true;
  }

  if(flags.gpu && (flags.storage_quad_precision || flags.calculation_quad_precision)){
    if(RANK == 0) std::cout << "Quad precision is only available with cpu.. exiting.. " << std::endl;
    exit(1);
  }
  
  int nov, nnz;
  string type;

  //std::string fname = &flags.filename[0];
  //This is to have filename in the struct, but ifstream don't like 
  //char*, so.
  //Type also goes same.
  //The reason they are being char* is they are also included in .cu
  //files
  
  FILE* f;
  int ret_code;
  MM_typecode matcode;
  int M, N, nz;
  int i;
  int *I, *J;
  
  if((f = fopen(flags.filename, "r")) == NULL){
    if(RANK==0) printf("Error opening the file, exiting.. \n");
    exit(1);
  }

  if(mm_read_banner(f, &matcode) != 0){
    if(RANK==0) printf("Could not process Matrix Market Banner, exiting.. \n");
    exit(1);
  }

  if(mm_is_matrix(matcode) != 1){
    if(RANK==0) printf("SUPerman only supports matrices, exiting.. \n");
    exit(1);
  }

  if(mm_is_coordinate(matcode) != 1){
    if(RANK==0) printf("SUPerman only supports mtx format at the moment, exiting.. \n");
    exit(1);
  }

  if((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0){
    if(RANK==0) printf("Matrix size cannot be read, exiting.. \n");
  }

  nnz = nz;

  if(M != N){
    if(RANK==0) printf("SUPerman only works with nxn matrices, exiting.. ");
    exit(1);
  }
  
  if(mm_is_complex(matcode) == 1){
    if(RANK==0) printf("SUPerman does not support complex type, exiting.. ");
    exit(1);
    //Instead of exit(1), there should be an escape function
    //which frees the allocated memory
  }
  
  bool is_pattern = false;
  if(mm_is_pattern(matcode) == 1)
    is_pattern = true;

  bool is_binary = false;
  if(flags.binary_graph)
    is_binary = true;

  bool is_symmetric = false;
  if(mm_is_symmetric(matcode) == 1 || mm_is_skew(matcode))
    is_symmetric = true;

  if(is_symmetric)
    nz *= 2;

  double final_perman;

  if(mm_is_real(matcode) == 1 && !flags.storage_half_precision && !is_pattern && !is_binary && !flags.storage_quad_precision){ 
    SparseMatrix<double>* sparsemat;
    DenseMatrix<double>* densemat;
    sparsemat = new SparseMatrix<double>();
    densemat = new DenseMatrix<double>(); 
    densemat->mat = new double[M*N];
    sparsemat->rvals = new double[nz];
    sparsemat->cvals = new double[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;

    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);

        
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
    
    print_flags(flags);

    bool scaling_chosen = 0;

    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;
    
    for(int i = 0; i < flags.rep; i++){

      Result result;
            
      DenseMatrix<double>* copy_densemat = copy_dense(densemat);
      SparseMatrix<double>* copy_sparsemat = copy_sparse(sparsemat);
      
      if(scaling_chosen){
	result = scale_and_calculate(copy_densemat, copy_sparsemat, flags, false);
	flags.type = "double"; //In case if scale_and_calculate change it
      }
      
      else{
	if(flags.compression)
	  result = compress_singleton_and_then_recurse(copy_densemat, copy_sparsemat, flags);
	else
	  result = RunAlgo(copy_densemat, copy_sparsemat, flags, false);
      }
      if(RANK==0) printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
      delete copy_densemat;
      delete copy_sparsemat;
    }
  }

 
#ifdef ONLYCPU
  
  else if(mm_is_real(matcode) == 1 && !flags.storage_half_precision && !is_pattern && !is_binary && flags.storage_quad_precision){
    SparseMatrix<__float128>* sparsemat;
    DenseMatrix<__float128>* densemat;
    sparsemat = new SparseMatrix<__float128>();
    densemat = new DenseMatrix<__float128>(); 
    densemat->mat = new __float128[M*N];
    sparsemat->rvals = new __float128[nz];
    sparsemat->cvals = new __float128[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;

    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
    
    print_flags(flags);

    bool scaling_chosen = 0;
    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;
    
    //No scaling for __float128
    //This is due to some standard library functions does not support __float128
    //and it already does not require scaling for accurate result
    
    for(int i = 0; i < flags.rep; i++){

      Result result;
      
      DenseMatrix<__float128>* copy_densemat = copy_dense(densemat);
      SparseMatrix<__float128>* copy_sparsemat = copy_sparse(sparsemat);
      
      if(flags.compression)
	result = compress_singleton_and_then_recurse(copy_densemat, copy_sparsemat, flags);
      else
	result = RunAlgo(copy_densemat, copy_sparsemat, flags, false);
      
      if(RANK==0) printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
      delete copy_densemat;
      delete copy_sparsemat;
    }
  }
  
#endif
  
  
  else if(mm_is_real(matcode) == 1 && flags.storage_half_precision && !is_pattern && !is_binary){
    SparseMatrix<float>* sparsemat;
    DenseMatrix<float>* densemat;
    sparsemat = new SparseMatrix<float>();
    densemat = new DenseMatrix<float>(); 
    densemat->mat = new float[M*N];
    sparsemat->rvals = new float[nz];
    sparsemat->cvals = new float[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;
    
    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
    print_flags(flags);

    bool scaling_chosen = 0;
    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;

    for(int i = 0; i < flags.rep; i++){
      Result result;

      DenseMatrix<float>* copy_densemat = copy_dense(densemat);
      SparseMatrix<float>* copy_sparsemat = copy_sparse(sparsemat);
      
      if(scaling_chosen){
	result = scale_and_calculate(copy_densemat, copy_sparsemat, flags, false);
	flags.type = "float";//In case if scale_and_calculate change it
      }

      else{
      
	if(flags.compression)
	  result = compress_singleton_and_then_recurse(copy_densemat, copy_sparsemat, flags);
	else
	  result = RunAlgo(copy_densemat, copy_sparsemat, flags, false);
	
      }
      if(RANK==0) printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
      delete copy_densemat;
      delete copy_sparsemat;
    }
  }
  
  else if(mm_is_integer(matcode) == 1 || is_pattern || is_binary){
    flags.type = "int";
    SparseMatrix<int>* sparsemat;
    DenseMatrix<int>* densemat;
    sparsemat = new SparseMatrix<int>();
    densemat = new DenseMatrix<int>();
    densemat->mat = new int[M*M];
    sparsemat->rvals = new int[nz];
    sparsemat->cvals = new int[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;    
    
    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);


    if(flags.preprocessing == 0)
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);

    print_flags(flags);
    
    bool scaling_chosen = 0;
        
    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;
    
    for(int i = 0; i < flags.rep; i++){
      Result result;

      DenseMatrix<int>* copy_densemat = copy_dense(densemat);
      SparseMatrix<int>* copy_sparsemat = copy_sparse(sparsemat);
      
      if(scaling_chosen){
	result = scale_and_calculate(copy_densemat, copy_sparsemat, flags, false);
	flags.type = "int";
      }
      
      else{
	if(flags.compression)
	  result = compress_singleton_and_then_recurse(copy_densemat, copy_sparsemat, flags);
	else
	  result = RunAlgo(copy_densemat, copy_sparsemat, flags, false);
      }
      if(RANK==0) printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
      delete copy_densemat;
      delete copy_sparsemat;
    }
  }
  
  else{
    if(RANK == 0) std::cout << "Matrix or flags have overlapping features.. " <<std::endl;
    print_flags(flags);
    exit(1);
  }

#ifdef MPIENABLED
  MPI_Finalize();
#endif
  
  return 0;
}
