//
// Created by deniz on 4/13/24.
//

#ifndef SUPERMAN_PERMANENT_H
#define SUPERMAN_PERMANENT_H

#include "omp.h"
#include "Matrix.h"
#include "Settings.h"
#include <algorithm>
#include "Result.h"
#include "IO.h"


template <class C, class S>
class Permanent
{
public:
    Permanent(Matrix<S>* matrix, Settings settings)
    :
        m_Matrix(matrix),
        m_Settings(settings) {}

    Permanent(const Permanent& other) = delete;
    Permanent& operator=(const Permanent& other) = delete;
    Permanent(Permanent&& other) = delete;
    Permanent& operator=(Permanent&& other) = delete;

    virtual ~Permanent()
    {
        delete m_Matrix;
    }

    Result computePermanent();
    virtual double permanentFunction() = 0;

public:
    Matrix<S>* m_Matrix;
    Settings m_Settings;
    
    double productSum;
};


template <class C, class S>
Result Permanent<C, S>::computePermanent()
{
    // cases for sparse
    if (m_Settings.algorithm == NAIVECODEGENERATION)
    {

    }
    else if (m_Settings.algorithm == REGEFFICIENTCODEGENERATION)
    {
        IO::UTOrder(m_Matrix);
    }
    else if (m_Settings.algorithm == APPROXIMATION)
    {
        IO::rowSort(m_Matrix);
        Matrix<S>* sparseMatrix = new SparseMatrix<S>(m_Matrix, getNNZ(m_Matrix));
        m_Matrix = sparseMatrix;
    }
    else if (m_Settings.algorithm != XREGISTERMSHARED && m_Settings.algorithm != XREGISTERMGLOBAL && m_Matrix->sparsity < 50)
    {
        IO::colSort(m_Matrix);
        Matrix<S>* sparseMatrix = new SparseMatrix<S>(m_Matrix, getNNZ(m_Matrix));
        m_Matrix = sparseMatrix;
    }

    double start = omp_get_wtime();
    double permanent = this->permanentFunction();
    double end = omp_get_wtime();

    Result result(end - start, permanent);

    return result;
}


#endif //SUPERMAN_PERMANENT_H
