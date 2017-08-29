// @(#)root/tmva/tmva/cnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing the Logistic Regression Layer                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Akshay Vashistha  <akshayvashistha1995@gmail.com>  - JSSATE, Noida, India *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *`
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/
#include "TestLogisticRegressionLayer.h"
#include "TMVA/DNN/Architectures/Cpu.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

int main()

{
   using Scalar_t = Double_t;
   std::cout << "Testing for Logistic Regression Layer started" << std::endl;

   testLayer1<TCpu<Scalar_t>>();

   return 0;
}
