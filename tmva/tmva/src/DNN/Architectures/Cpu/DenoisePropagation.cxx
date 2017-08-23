// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha(ajatgd)

/*************************************************************************
 * Copyright (C) 2017 ajatgd                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the Denoise Autoencoder functions for the  //
 // reference implementation.                                    //
//////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMVA/DNN/Architectures/Cpu/Blas.h"
#include <cassert>

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::AddBiases(TCpuMatrix<AFloat> &B,
                             const TCpuMatrix<AFloat> &biases)
{
   int m = (int)B.GetNrows();
   int n = (int)B.GetNcols();

   int inc = 1.0;
   AFloat alpha = 1.0;

   AFloat *A = B.GetRawDataPointer();
   const AFloat *x = biases.GetRawDataPointer();
   const AFloat *y = TCpuMatrix<AFloat>::GetOnePointer();

   ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);

}

//______________________________________________________________________________

template <typename AFloat>
void TCpu<AFloat>::UpdateParams(
   TCpuMatrix<AFloat> &x, TCpuMatrix<AFloat> &tildeX, TCpuMatrix<AFloat> &y,
   TCpuMatrix<AFloat> &z, TCpuMatrix<AFloat> &fVBiases, TCpuMatrix<AFloat> &fHBiases,
   TCpuMatrix<AFloat> &fWeights, TCpuMatrix<AFloat> &VBiasError,
   TCpuMatrix<AFloat> &HBiasError, AFloat learningRate, size_t fBatchSize)
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

template<typename AFloat>
void TCpu<AFloat>::SoftmaxAE(TCpuMatrix<AFloat> & A)
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

template <typename AFloat>
void TCpu<AFloat>::CorruptInput(TCpuMatrix<AFloat> &input,
                                      TCpuMatrix<AFloat> &corruptedInput,
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

template <typename AFloat>
void TCpu<AFloat>::EncodeInput(TCpuMatrix<AFloat> &input,
                                     TCpuMatrix<AFloat> &compressedInput,
                                     TCpuMatrix<AFloat> &Weights)
{
   size_t m, a;
   m = compressedInput.GetNrows();
   a = input.GetNrows();

   for (size_t i = 0; i < m; i++) {
      compressedInput(i, 0) = 0;
      for (size_t j = 0; j < a; j++) {
         compressedInput(i, 0) =
         compressedInput(i, 0) + (Weights(i, j) * input(j, 0));
      }
   }
}
//______________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::ReconstructInput(TCpuMatrix<AFloat> &compressedInput,
                                          TCpuMatrix<AFloat> &reconstructedInput,
                                          TCpuMatrix<AFloat> &fWeights) {

   for (size_t i=0; i<(size_t)reconstructedInput.GetNrows(); i++)
   {
      reconstructedInput(i, 0) = 0;
      for(size_t j=0; j<(size_t)compressedInput.GetNrows();j++)
      {
         reconstructedInput(i, 0) += fWeights(j, i) * compressedInput(j, 0);
      }
   }
}

//______________________________________________________________________________
// Logistic Regression Layer Methods
//
//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::ForwardLogReg(TCpuMatrix<AFloat> &input,
                                       TCpuMatrix<AFloat> &p,
                                       TCpuMatrix<AFloat> &fWeights)
{
   size_t m,n;
   m = p.GetNrows();
   n = input.GetNrows();
   for(size_t i= 0; i < m; i++)
   {
      p(i, 0) = 0;
      for(size_t j=0; j < n; j++)
      {
         p(i, 0) += fWeights(i, j) * input(j, 0);
      }
   }
}

//______________________________________________________________________________

template<typename AFloat>
void TCpu<AFloat>::UpdateParamsLogReg(TCpuMatrix<AFloat> &input,
                                            TCpuMatrix<AFloat> &output,
                                            TCpuMatrix<AFloat> &difference,
                                            TCpuMatrix<AFloat> &p,
                                            TCpuMatrix<AFloat> &fWeights,
                                            TCpuMatrix<AFloat> &fBiases,
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
