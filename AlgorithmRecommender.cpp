//
// Created by deniz on 3/14/24.
//

#include "AlgorithmRecommender.h"


Recommendation AlgorithmRecommender::m_Recommendation;

Recommendation AlgorithmRecommender::recommendAlgorithm(const char *filename, bool onGPU, int RANK)
{
    AlgorithmRecommender::analyzeFile(filename, RANK);
    AlgorithmRecommender::isBinaryFile(filename);
    AlgorithmRecommender::determineAlgorithm(onGPU);

    return AlgorithmRecommender::AlgorithmRecommender::m_Recommendation;
}

void AlgorithmRecommender::analyzeFile(const char* filename, int RANK)
{
    FILE* f;
    MM_typecode matcode;
    int ret_code;
    int M, N, nnz;

    if((f = fopen(filename, "r")) == NULL)
    {
        if(RANK==0) printf("Error opening the file, exiting.. \n");
        exit(1);
    }

    if(mm_read_banner(f, &matcode) != 0)
    {
        if(RANK==0) printf("Could not process Matrix Market Banner, exiting.. \n");
        exit(1);
    }

    if (mm_is_real(matcode) != 1)
    {
        // CHECK THIS PART OUT LATER
        exit(1);
    }

    if(mm_is_matrix(matcode) != 1)
    {
        if(RANK==0) printf("SUPerman only supports matrices, exiting.. \n");
        exit(1);
    }

    if(mm_is_coordinate(matcode) != 1)
    {
        if(RANK==0) printf("SUPerman only supports mtx format at the moment, exiting.. \n");
        exit(1);
    }

    if((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nnz)) != 0)
    {
        if(RANK==0) printf("Matrix size cannot be read, exiting.. \n");
        exit(1);
    }

    if(M != N)
    {
        if(RANK==0) printf("SUPerman only works with nxn matrices, exiting.. ");
        exit(1);
    }

    if(mm_is_complex(matcode) == 1)
    {
        if(RANK==0) printf("SUPerman does not support complex type, exiting.. ");
        exit(1);
    }

    if(mm_is_pattern(matcode) == 1)
        AlgorithmRecommender::m_Recommendation.isPatternSymmetric = true;
    else
        AlgorithmRecommender::m_Recommendation.isPatternSymmetric = false;

    if(mm_is_symmetric(matcode) == 1 || mm_is_skew(matcode))
    {
        AlgorithmRecommender::m_Recommendation.isSymmetric = true;
        nnz *= 2;
    }
    else
    {
        AlgorithmRecommender::m_Recommendation.isSymmetric = false;
    }

    AlgorithmRecommender::m_Recommendation.M = M;
    AlgorithmRecommender::m_Recommendation.N = N;
    AlgorithmRecommender::m_Recommendation.nnz = nnz;

    fclose(f);

    if (isBinaryFile(filename))
    {
        AlgorithmRecommender::m_Recommendation.isBinary = true;
    }
    else
    {
        AlgorithmRecommender::m_Recommendation.isBinary = false;
    }
}

bool AlgorithmRecommender::isBinaryFile(const char *filename)
{
    std::ifstream file(filename);

    while (file.peek() == '%') file.ignore(2048, '\n');

    int row, col, line;
    file >> row >> col >> line;

    std::string firstLine;
    std::getline(file, firstLine);
    int spaceCount = 0;
    for (const auto& c: firstLine)
    {
        if (c == ' ')
        {
            ++spaceCount;
        }
    }
    if (spaceCount == 1)
    {
        return false;
    }

    int value;
    for (int i = 1; i < line; ++i)
    {
        file >> value >> value >> value;
        if (value != 0 && value != 1)
        {
            return false;
        }
    }
    return true;
}

void AlgorithmRecommender::determineAlgorithm(bool onGPU)
{
    // ASK THE CORRECTNESS OF THE FOLLOWING ALGORITHM DETERMINATION
    // ALSO ASK INTO WHERE OTHER ALGORITHMS SHOULD BE INJECTED (RASMUSSEN, SKIP ORDER BALANCED etc.)

    double sparsity = (AlgorithmRecommender::m_Recommendation.nnz / (AlgorithmRecommender::m_Recommendation.M * AlgorithmRecommender::m_Recommendation.N)) * 100;

    if (onGPU)
    {
        if (sparsity < 50)
        {
            AlgorithmRecommender::m_Recommendation.algorithm = ParallelPermanSortOrder;
            AlgorithmRecommender::m_Recommendation.type = Sparse;
        }
        else
        {
            AlgorithmRecommender::m_Recommendation.algorithm = ParallelPerman;
            AlgorithmRecommender::m_Recommendation.type = Dense;
        }
        return;
    }

    if (sparsity < 30)
    {
        AlgorithmRecommender::m_Recommendation.algorithm = ParallelPermanSkipOrder;
        AlgorithmRecommender::m_Recommendation.type = Sparse;
    }
    else if (sparsity > 30 && sparsity < 50)
    {
        AlgorithmRecommender::m_Recommendation.algorithm = ParallelPermanSortOrder;
        AlgorithmRecommender::m_Recommendation.type = Sparse;
    }
    else
    {
        AlgorithmRecommender::m_Recommendation.algorithm = ParallelPerman;
        AlgorithmRecommender::m_Recommendation.type = Dense;
    }
}
