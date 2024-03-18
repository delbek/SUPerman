//
// Created by deniz on 3/14/24.
//

#ifndef SUPERMAN_ALGORITHMRECOMMENDER_H
#define SUPERMAN_ALGORITHMRECOMMENDER_H

#include <vector>
#include <string>
#include <fstream>
#include "MatrixMarketIOLibrary.h"
#include <stdio.h>


enum MatrixTypes
{
    Sparse,
    Dense
};

enum Algorithms
{
    ParallelPerman,
    ParallelPermanSkipOrder,
    ParallelPermanSkipOrderBalanced,
    ParallelPermanSortOrder,
    Rasmussen
};

struct Recommendation
{
    MatrixTypes type;
    Algorithms algorithm;

    int M;
    int N;
    int nnz;

    bool isPatternSymmetric;
    bool isSymmetric;
    bool isBinary;
};


class AlgorithmRecommender
{
public:
    static Recommendation recommendAlgorithm(const char* filename, bool onGPU, int RANK);

private:
    static void analyzeFile(const char* filename, int RANK);
    static bool isBinaryFile(const char* filename);
    static void determineAlgorithm(bool onGPU);

private:
    static Recommendation m_Recommendation;

};


#endif //SUPERMAN_ALGORITHMRECOMMENDER_H
