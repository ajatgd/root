// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableDAETransform                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Principal value composition of input variables                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *    Marc Huwiler    <marc.huwiler@windowslive.com> - CERN, Switzerland          *
 *    Akshay Vashistha <akshayvashistha1995@gmail.com> - JSSATE, Noida, India     *
 *                                                                                *
 * Copyright (c) 2017:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableDAETransform
#define ROOT_TMVA_VariableDAETransform

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableDAETransform                                                 //
//                                                                      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//#include "TPrincipal.h"

#include "TMVA/VariableTransformBase.h"
//#include "TMVA/DNN/DAE2/CompressionLayer.h"
#include "TMVA/DNN/DeepNet.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/DAE/CompressionLayer.h"
#include "TMVA/DNN/DAE/CorruptionLayer.h"
#include "TMVA/DNN/DAE/ReconstructionLayer.h"
#include "TMVA/DNN/DeepNet.h"

//using namespace TMVA::DNN;
//using namespace TMVA::DNN::DAE;

namespace TMVA {
   //namespace DNN {

   class VariableDAETransform : public VariableTransformBase {

   public:
      using Architecture_t = DNN::TReference<Double_t>;
      using Matrix_t = typename Architecture_t::Matrix_t;
      using Scalar_t = typename Architecture_t::Scalar_t;

      VariableDAETransform( DataSetInfo& dsi );
      virtual ~VariableDAETransform( void );

      void   Initialize();
      //template <typename Architecture_t> void   Initialize();
      Bool_t PrepareTransformation (const std::vector<Event*>&);

      //Bool_t GetEventValues( const Event* event, Matrix_t& input, std::vector<Char_t>& mask, Bool_t backTransform = kFALSE  ) const;; 

      virtual const Event* Transform(const Event* const, Int_t cls ) const;
      virtual const Event* InverseTransform(const Event* const, Int_t cls ) const;

      void WriteTransformationToStream ( std::ostream& ) const;
      void ReadTransformationFromStream( std::istream&, const TString& );

      virtual void AttachXMLTo(void* parent);
      virtual void ReadFromXML( void* trfnode );

      // writer of function code
      virtual void MakeFunction( std::ostream& fout, const TString& fncName, Int_t part, UInt_t trCounter, Int_t cls );

   private:

      void TrainOnExampleData( const std::vector< Event*>& );

      void TransformInputDataset(const std::vector<Event*>&, std::vector<Matrix_t>&);     //Maybe rename into Convert  
      void TransformInputData( const std::vector<Float_t>& localEvent, Matrix_t& remoteInputs) const; 
      void BackTransformOutputData(const Matrix_t&, std::vector<Float_t>&) const; 

      void X2P( std::vector<Float_t>&, const std::vector<Float_t>&, Int_t cls ) const;
      void P2X( std::vector<Float_t>&, const std::vector<Float_t>&, Int_t cls ) const;

      std::vector<DNN::TDeepAutoEncoder<Architecture_t>* > fAutoEncoder; 
      //TCompressionLayer fEncoder; 

      std::vector<Matrix_t>* input; 
      std::vector<Matrix_t>* output; 

      
      // store relevant parts of PCA locally
      std::vector<TVectorD*> fMeanValues;   // mean values
      std::vector<TMatrixD*> fEigenVectors; // eigenvectors

      ClassDef(VariableDAETransform,0); // Variable transformation: Principal Value Composition
   };

  //} // namespace DNN 

} // namespace TMVA

#endif
