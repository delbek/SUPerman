//
// Created by delbek on 6/6/24.
//

#ifndef SUPERMAN_DECOMPOSEPERMAN_H
#define SUPERMAN_DECOMPOSEPERMAN_H


#include "Permanent.h"
#include "omp.h"
#include "Matrix.h"
#include "Settings.h"
#include <algorithm>
#include "Result.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include "IO.h"
#include <vector>
#include "Helpers.h"


template <class C, class S, class Permanent>
class DecomposePerman
{
public:
    DecomposePerman(Matrix<S>* matrix, Settings settings)
    :
        m_Matrix(matrix),
        m_Settings(settings) {}

    DecomposePerman(const DecomposePerman& other) = delete;
    DecomposePerman& operator=(const DecomposePerman& other) = delete;
    DecomposePerman(DecomposePerman&& other) = delete;
    DecomposePerman& operator=(DecomposePerman&& other) = delete;

    Result computePermanentRecursively();

private:
    bool compress1NNZ(Matrix<S>* matrix);
    bool compress2NNZ(Matrix<S>* matrix);
    bool compress34NNZ(Matrix<S>* matrix1, Matrix<S>* matrix2, int minDeg);

    void startRecursion(Matrix<S>* matrix);
    void recurse(Matrix<S>* matrix);
    void addQueue(Matrix<S>* matrix);

protected:
    Matrix<S>* m_Matrix;
    Settings m_Settings;
    std::vector<Permanent*> m_Permanents;
    std::vector<ScalingCompact*> m_ScalingValues;

    unsigned m_1Decompose;
    unsigned m_2Decompose;
    unsigned m_34Decompose;
};


template <class C, class S, class Permanent>
Result DecomposePerman<C, S, Permanent>::computePermanentRecursively()
{
    m_1Decompose = 0;
    m_2Decompose = 0;
    m_34Decompose = 0;

    double start = omp_get_wtime();

    startRecursion(m_Matrix);

    double overall = 0;
    if (m_Permanents.size() > 1)
    {
        std::stringstream stream;
        stream << "The computation of the original permanent is partitioned into the computation of the " << m_Permanents.size() << " sub-permanent." << std::endl;
        print(stream, this->m_Settings.rank, this->m_Settings.PID, -1);
    }
    for (int p = 0; p < m_Permanents.size(); ++p)
    {
        auto derived = dynamic_cast<Permanent*>(m_Permanents[p]);
        double result;
        if (m_Settings.algorithm != APPROXIMATION)
        {
            result = ((4 * (derived->m_Matrix->nov % 2) - 2) * derived->productSum);
        }
        else
        {
            result = derived->productSum;
        }
        if (m_Settings.scaling)
        {
            auto scalingCompact = m_ScalingValues[p];
            double* rowScale = scalingCompact->rowScale;
            double* colScale = scalingCompact->colScale;
            for (int i = 0; i < derived->m_Matrix->nov; ++i)
            {
                result /= rowScale[i];
                result /= colScale[i];
            }
            delete scalingCompact;
        }
        overall += result;
        delete derived;
    }

    std::stringstream stream;
    stream << "Number of 1 NNZ decompositions performed: " << m_1Decompose << std::endl;
    stream << "Number of 2 NNZ decompositions performed: " << m_2Decompose << std::endl;
    stream << "Number of 3-4 NNZ decompositions performed: " << m_34Decompose << std::endl;
    print(stream, this->m_Settings.rank, this->m_Settings.PID, -1);

    double end = omp_get_wtime();
    Result result(end - start, overall);

    return result;
}

template <class C, class S, class Permanent>
void DecomposePerman<C, S, Permanent>::startRecursion(Matrix<S>* matrix)
{
    if (this->m_Settings.algorithm == APPROXIMATION)
    {
        addQueue(matrix);
        return;
    }

    bool isCompressed = true;
    while (isCompressed && matrix->nov > 30)
    {
        isCompressed = compress1NNZ(matrix);

        if (!isCompressed)
        {
            isCompressed = compress2NNZ(matrix);
        }

        if (isCompressed && isRankDeficient(matrix))
        {
            std::stringstream stream;
            stream << "Matrix is rank deficient." << std::endl;
            print(stream, this->m_Settings.rank, this->m_Settings.PID, -1);
            return;
        }
    }

    if (this->m_Settings.algorithm == NAIVECODEGENERATION || this->m_Settings.algorithm == REGEFFICIENTCODEGENERATION)
    {
        addQueue(matrix);
        return;
    }

    recurse(matrix);
}

template <class C, class S, class Permanent>
void DecomposePerman<C, S, Permanent>::recurse(Matrix<S>* matrix)
{
    int minDeg = getMinDegree(matrix);
    if (minDeg < 5 && matrix->nov > 30)
    {
        if (minDeg == 1)
        {
            compress1NNZ(matrix);
            recurse(matrix);
        }
        else if (minDeg == 2)
        {
            compress2NNZ(matrix);
            recurse(matrix);
        }
        else if (minDeg == 3 || minDeg == 4)
        {
            auto matrix2 = new Matrix<S>;
            compress34NNZ(matrix, matrix2, minDeg);
            recurse(matrix);
            recurse(matrix2);
            delete matrix2;
        }
    }
    else
    {
        addQueue(matrix);
    }
}

template <class C, class S, class Permanent>
void DecomposePerman<C, S, Permanent>::addQueue(Matrix<S> *matrix)
{
    int nnz = getNNZ(matrix);
    matrix->sparsity = (double(nnz) / double(matrix->nov * matrix->nov)) * 100;

    Matrix<S>* newMatrix = new Matrix<S>(*matrix);
    if (newMatrix->nov > 63 && m_Settings.algorithm != APPROXIMATION)
    {
        throw std::runtime_error("Permanent is an #P-complete problem. The size of the matrix you want to calculate the permanent for exceeds the limit of what is computationally possible. Try approximation algorithms.\n");
    }
    if (m_Settings.scaling)
    {
        ScalingCompact* scalingCompact = new ScalingCompact;
        IO::scale<C, S>(newMatrix, m_Settings, scalingCompact);
        m_ScalingValues.push_back(scalingCompact);
    }
    Permanent* newPermanent = new Permanent(newMatrix, m_Settings);
    newPermanent->computePermanent();
    m_Permanents.push_back(newPermanent);
}

template <class C, class S, class Permanent>
bool DecomposePerman<C, S, Permanent>::compress1NNZ(Matrix<S> *matrix)
{
    S*& mat = matrix->mat;
    int& nov = matrix->nov;

    int row = -1;
    int col = -1;
    // is there a row or col of degree 1
    for (int i = 0; i < nov; ++i)
    {
        if (getRowNNZ(matrix, i) == 1)
        {
            row = i;
            break;
        }
        if (getColNNZ(matrix, i) == 1)
        {
            col = i;
            break;
        }
    }

    if (row == -1 && col == -1)
    {
        return false;
    }

    ++m_1Decompose;

    S val;
    if (row != -1) // if there is a row of degree 1
    {
        for (int j = 0; j < nov; ++j)
        {
            if (mat[row * nov + j] != 0)
            {
                val = mat[row * nov + j];
                col = j;
                break;
            }
        }
    }
    else // if there is a col of degree 1
    {
        for (int i = 0; i < nov; ++i)
        {
            if (mat[i * nov + col] != 0)
            {
                val = mat[i * nov + col];
                row = i;
                break;
            }
        }
    }

    // compressing a.k.a removing row and col that this nonzero belongs to
    S* compressedMat = new S[nov * nov];
    for (int i = 0; i < nov; ++i)
    {
        for (int j = 0; j < nov; ++j)
        {
            if (i != row && j != col)
            {
                S entry = mat[i * nov + j];
                int currentRow = i;
                if (i > row)
                {
                    --currentRow;
                }

                int currentCol = j;
                if (j > col)
                {
                    --currentCol;
                }

                compressedMat[currentRow * (nov - 1) + currentCol] = entry;
            }
        }
    }

    --nov;
    memcpy(mat, compressedMat, sizeof(S) * nov * nov);

    for (int i = 0; i < nov; ++i)
    {
        mat[i] *= val; // multiplying each row with the value of the nonzero that cease to exist now
    }

    delete[] compressedMat;

    return true;
}

template <class C, class S, class Permanent>
bool DecomposePerman<C, S, Permanent>::compress2NNZ(Matrix<S>* matrix)
{
    S*& mat = matrix->mat;
    int& nov = matrix->nov;

    int row = -1;
    int col = -1;

    // is there a row or col of degree 2
    for (int i = 0; i < nov; ++i)
    {
        if (getRowNNZ(matrix, i) == 2)
        {
            row = i;
            break;
        }
        if (getColNNZ(matrix, i) == 2)
        {
            col = i;
            break;
        }
    }

    if (row == -1 && col == -1)
    {
        return false;
    }

    ++m_2Decompose;

    int firstNNZ = -1;
    int secondNNZ = -1;
    if (row != -1)  // if there is a row of degree 2
    {
        for (int j = 0; j < nov; ++j)
        {
            if (mat[row * nov + j] != 0)
            {
                if (firstNNZ == -1)
                {
                    firstNNZ = j;
                }
                else
                {
                    secondNNZ = j;
                    break;
                }
            }
        }
    }
    else // if there is a col of degree 2
    {
        for (int i = 0; i < nov; ++i)
        {
            if (mat[i * nov + col] != 0)
            {
                if (firstNNZ == -1)
                {
                    firstNNZ = i;
                }
                else
                {
                    secondNNZ = i;
                    break;
                }
            }
        }
    }

    S* compressedMat = new S[nov * nov];
    if (row != -1) // deleting the row in which 2 nonzero located
    {
        for (int i = 0; i < nov; ++i)
        {
            for (int j = 0; j < nov; ++j)
            {
                if (i != row && j != secondNNZ)
                {
                    S entry = mat[i * nov + j];

                    int currentRow = i;
                    if(i > row)
                    {
                        --currentRow;
                    }
                    int currentCol = j;
                    if (j > secondNNZ)
                    {
                        --currentCol;
                    }

                    if (j == firstNNZ)
                    {
                        entry = mat[i * nov + firstNNZ] * mat[row * nov + secondNNZ] + mat[i * nov + secondNNZ] * mat[row * nov + firstNNZ];

                    }

                    compressedMat[currentRow * (nov - 1) + currentCol] = entry;
                }
            }
        }
    }
    else // deleting the col in which 2 nonzero located
    {
        for (int i = 0; i < nov; ++i)
        {
            for (int j = 0; j < nov; ++j)
            {
                if (i != secondNNZ && j != col)
                {
                    S entry = mat[i * nov + j];

                    int currentRow = i;
                    if (i > secondNNZ)
                    {
                        --currentRow;
                    }
                    int currentCol = j;
                    if (j > col)
                    {
                        --currentCol;
                    }

                    if (i == firstNNZ)
                    {
                        entry = mat[firstNNZ * nov + j] * mat[secondNNZ * nov + col] + mat[secondNNZ * nov + j] * mat[firstNNZ * nov + col];
                    }

                    compressedMat[currentRow * (nov - 1) + currentCol] = entry;
                }
            }
        }
    }

    --nov;
    memcpy(mat, compressedMat, sizeof(S) * nov * nov);

    delete[] compressedMat;
    return true;
}

template <class C, class S, class Permanent>
bool DecomposePerman<C, S, Permanent>::compress34NNZ(Matrix<S>* matrix1, Matrix<S>* matrix2, int minDeg)
{
    S*& mat = matrix1->mat;
    S*& mat2 = matrix2->mat;
    int& nov = matrix1->nov;
    int& nov2 = matrix2->nov;

    int row = -1;
    int col = -1;

    for (int i = 0; i < nov; ++i)
    {
        if (getRowNNZ(matrix1, i) == minDeg)
        {
            row = i;
            break;
        }
        if (getColNNZ(matrix1, i) == minDeg)
        {
            col = i;
            break;
        }
    }

    if (row == -1 && col == -1)
    {
        return false;
    }

    ++m_34Decompose;

    S* transposeMatrix = new S[nov * nov];
    if (row == -1)
    {
        for (int i = 0; i < nov; ++i)
        {
            for (int j = 0; j < nov; ++j)
            {
                transposeMatrix[j * nov + i] = mat[i * nov + j];
            }
        }
        row = col;
    }
    else
    {
        memcpy(transposeMatrix, mat, sizeof(S) * nov * nov);
    }

    int nonzeros[4] = {-1, -1, -1, -1};
    int index = 0;
    int zeroLocation = -1;
    for (int j = 0; j < nov; ++j)
    {
        if (transposeMatrix[row * nov + j] != 0)
        {
            nonzeros[index++] = j;
        }
        else
        {
            zeroLocation = j;
        }
    }

    if (nonzeros[3] == -1)
    {
        nonzeros[3] = zeroLocation;
    }

    mat2 = new S[nov * nov];
    memset(mat, 0, sizeof(S) * nov * nov);
    memset(mat2, 0, sizeof(S) * nov * nov);

    for (int i = 0; i < nov; ++i)
    {
        if (i != row)
        {
            int currentRow = i;
            if (i > row)
            {
                --currentRow;
            }

            for (int j = 0; j < nov; ++j)
            {
                if(j != nonzeros[1])
                {
                    int currentCol = j;
                    if (j > nonzeros[1])
                    {
                        --currentCol;
                    }
                    if (j != nonzeros[0])
                    {
                        mat[currentRow * (nov - 1) + currentCol] = transposeMatrix[i * nov + j];
                    }
                    else
                    {
                        mat[currentRow * (nov - 1) + currentCol] = transposeMatrix[row * nov + nonzeros[0]] * transposeMatrix[i * nov + nonzeros[1]] + transposeMatrix[row * nov + nonzeros[1]] * transposeMatrix[i * nov + nonzeros[0]];
                    }
                }

                if (j != nonzeros[3])
                {
                    int currentCol = j;
                    if (j > nonzeros[3])
                    {
                        --currentCol;
                    }
                    if (j != nonzeros[2])
                    {
                        mat2[currentRow * (nov - 1) + currentCol] = transposeMatrix[i * nov + j];
                    }
                    else
                    {
                        mat2[currentRow * (nov - 1) + currentCol] = transposeMatrix[row * nov + nonzeros[2]] * transposeMatrix[i * nov + nonzeros[3]] + transposeMatrix[row * nov + nonzeros[3]] * transposeMatrix[i * nov + nonzeros[2]];
                    }
                }
            }
        }
    }

    --nov;
    nov2 = nov;
    delete[] transposeMatrix;
    return true;
}


#endif //SUPERMAN_DECOMPOSEPERMAN_H
