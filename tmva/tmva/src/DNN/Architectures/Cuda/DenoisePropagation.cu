// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *   Implementation of the Autoencoder functions for the CUDA implementation.     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Akshay Vashistha <akshayvashistha1995@gmail.com>  - JSSATE, Noida, India  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"
#include <cassert>

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::AddBiases(TCudaMatrix<AFloat> &A,
                              const TCudaMatrix<AFloat> &biases)
{
  dim3 blockDims = TDevice::BlockDims2D();
  dim3 gridDims  = TDevice::GridDims2D(A);
  cudaStream_t s = A.GetComputeStream();
  ::TMVA::DNN::Cuda::AddRowWise<<<gridDims, blockDims, 0, s>>>(
      A.GetDataPointer(),
      biases.GetDataPointer(),
      A.GetNrows(),
      A.GetNcols());
}

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::UpdateParams(TCudaMatrix<AFloat> &x,
                                TCudaMatrix<AFloat> &z,
                                TCudaMatrix<AFloat> &fVBiases,
                                TCudaMatrix<AFloat> &fHBiases,
                                TCudaMatrix<AFloat> &fWeights,
                                AFloat learningRate,
                                AFloat corruptionLevel,
                                size_t fBatchSize)
{
   //updating fVBiases
   for (size_t i = 0; i < (size_t)fVBiases.GetNrows(); i++)
   {
      for (size_t j = 0; j < (size_t)fVBiases.GetNcols(); j++) {
         VBiasError(i, j) = x(i, j) - z(i, j);
         fVBiases(i, j) += learningRate * VBiasError(i, j) / fBatchSize;
      }
   }

   //updating fHBiases
   for(size_t i = 0; i < fHBiases.GetNrows(); i++)
   {
      HBiasError(i,0) = 0;
      for(size_t j = 0; j < fVBiases.GetNrows(); j++)
      {
         HBiasError(i, 0) += fWeights(i, j) * VBiasError(j, 0);
      }
      HBiasError(i, 0) *= y(i, 0) * (1 - y(i, 0));
      fHBiases(i, 0) += learningRate * HBiasError(i, 0) / fBatchSize;
   }

   //updating weights
   for(size_t i = 0; i < fHBiases.GetNrows(); i++)
   {
      for(size_t j = 0; j< fVBiases.GetNrows(); j++)
      {
         fWeights(i, j) += learningRate * (HBiasError(i, 0) * tildeX(j, 0) +
                                        VBiasError(j, 0) * y(i, 0)) / fBatchSize;
      }
   }

}
//______________________________________________________________________________

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::SoftmaxAE(TCudaMatrix<AFloat> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   Real_t sum = 0.0;
   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         sum += exp(A(i, j));
      }
   }

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i, j) = exp(A(i, j)) / sum;
      }
   }
}

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::CorruptInput(TCudaMatrix<AFloat> & input,
                                 TCudaMatrix<AFloat> & corruptedInput,
                                 AFloat corruptionLevel)
{
   assert(corruptionLevel != 0 && corruptionLevel != 1 && "corruptionLevel cannot be 0 or 1");
   for(size_t i=0; i< (size_t)input.GetNrows(); i++)
   {
      for(size_t j=0; j<(size_t)input.GetNcols(); j++ )
      {

         if ((size_t)((rand() / (RAND_MAX + 1.0)) * 100) %
             ((size_t)(corruptionLevel * 10)) ==
         0)
         {
            corruptedInput(i, j) = 0;
         }
         else
         {
            corruptedInput(i, j) = input(i, j);
         }
      }
   }
}


//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::EncodeInput(TCudaMatrix<AFloat> & input,
                                TCudaMatrix<AFloat> & compressedInput,
                                TCudaMatrixT<AFloat> &fWeights)
{
   Multiply(compressedInput, Weights, input);
}
//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ReconstructInput(TCudaMatrix<AFloat> & compressedInput,
                                     TCudaMatrix<AFloat> & reconstructedInput,
                                     TCudaMatrix<AFloat> &fWeights)
{
   Multiply(reconstructedInput, fWeights, compressedInput);
}

//______________________________________________________________________________
// Logistic Regression Layer Methods
//
//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::ForwardLogReg(TCudaMatrix<AFloat> &input,
                                  TCudaMatrix<AFloat> &p,
                                  TCudaMatrix<AFloat> &fWeights)
{
   Multiply(p,fWeights,input);
}

//______________________________________________________________________________

template<typename AFloat>
void TCuda<AFloat>::UpdateParamsLogReg(TCudaMatrix<AFloat> &input,
                                       TCudaMatrix<AFloat> &output,
                                       TCudaMatrix<AFloat> &difference,
                                       TCudaMatrix<AFloat> &p,
                                       TCudaMatrix<AFloat> &fWeights,
                                       TCudaMatrixT<AFloat> &fBiases,
                                       AFloat learningRate,
                                       size_t fBatchSize)
{
   size_t m,n;
   m = p.GetNrows();
   n = input.GetNrows();

   for(size_t i= 0; i<m; i++)
   {
      difference(i, 0) = output(i, 0) - p(i, 0);
      for(size_t j=0; j<n; j++)
      {
         fWeights(i, j) +=
         learningRate * difference(i, 0) * input(j, 0) / fBatchSize;
      }

      fBiases(i, 0) += learningRate * difference(i, 0) / fBatchSize;
   }
}

//______________________________________________________________________________


}
}
