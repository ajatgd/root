// @(#)root/tmva $Id: MethodHMatrix.h,v 1.3 2006/08/31 11:03:37 rdm Exp $
// Author: Andreas Hoecker, Xavier Prudent, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodHMatrix                                                         *
 *                                                                                *
 * Description:                                                                   *
 *      H-Matrix method, which is implemented as a simple comparison of           *
 *      chi-squared estimators for signal and background, taking into account     *
 *      the linear correlations between the input variables.                      *
 *      Method is (also) used by D0 Collaboration (FNAL) for electron             *
 *      identification; for more information, see, eg,                            *
 *      http://www-d0.fnal.gov/d0dist/dist/packages/tau_hmchisq/devel/doc/        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany                                                *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodHMatrix
#define ROOT_TMVA_MethodHMatrix

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodHMatrix                                                        //
//                                                                      //
// H-Matrix method, which is implemented as a simple comparison of      //
// chi-squared estimators for signal and background, taking into        //
// account the linear correlations between the input variables          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMVA_TVectorD
#include "TVectorD.h"
#endif

namespace TMVA {

   class MethodHMatrix : public MethodBase {

   public:

      MethodHMatrix( TString jobName,
                     std::vector<TString>* theVariables,
                     TTree* theTree = 0,
                     TString theOption = "",
                     TDirectory* theTargetDir = 0 );

      MethodHMatrix( std::vector<TString> *theVariables,
                     TString theWeightFile,
                     TDirectory* theTargetDir = NULL );

      virtual ~MethodHMatrix( void );

      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToFile( void );

      // read weights from file
      virtual void ReadWeightsFromFile( void );

      // calculate the MVA value
      virtual Double_t GetMvaValue( Event *e );

      // write method specific histos to target file
      virtual void WriteHistosToFile( void );

   protected:

   private:

      // returns chi2 estimator for given type (signal or background)
      Double_t GetChi2( Event *e, Type ) const;

      // arrays of input evt vs. variable
      TMatrixD* fInvHMatrixS; // inverse H-matrix (signal)
      TMatrixD* fInvHMatrixB; // inverse H-matrix (background)
      TVectorD* fVecMeanS;    // vector of mean values (signal)
      TVectorD* fVecMeanB;    // vector of mean values (background)

      Bool_t    fNormaliseInputVars; // normalise input variables

      // default initialisation method called by all constructors
      void InitHMatrix( void );

      ClassDef(MethodHMatrix,0) // H-Matrix method, a simple comparison of chi-squared estimators for signal and background
   };

} // namespace TMVA

#endif
