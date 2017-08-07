// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableAutoEncoderTransform                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Variable Transformation using Deep AutoEncoders                           *
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

#ifndef ROOT_TMVA_VARIABLE_AUTOENCODER_HANDLER
#define ROOT_TMVA_VARIABLE_AUTOENCODER_HANDLER

#ifndef ROOT_TPrincipal
#include "TPrincipal.h"
#endif

#ifndef ROOT_TMVA_VariableTransformBase
#include "TMVA/VariableTransformBase.h"
#endif

namespace TMVA {

   class VariableAutoEncoderTransform : public VariableTransformBase {

   public:

      VariableAutoEncoderTransform( DataSetInfo& dsi );
      virtual ~VariableAutoEncoderTransform( void );

      void   Initialize();
      Bool_t PrepareTransformation (const std::vector<Event*>&);

      virtual const Event* Transform(const Event* const, Int_t cls ) const;
      virtual const Event* InverseTransform(const Event* const, Int_t cls ) const;

      void WriteTransformationToStream ( std::ostream& ) const;
      void ReadTransformationFromStream( std::istream&, const TString& );

      virtual void AttachXMLTo(void* parent);
      virtual void ReadFromXML( void* trfnode );

      // writer of function code
      virtual void MakeFunction( std::ostream& fout, const TString& fncName, Int_t part, UInt_t trCounter, Int_t cls );

   private:

      void CalculatePrincipalComponents( const std::vector< Event*>& );
      void X2P( std::vector<Float_t>&, const std::vector<Float_t>&, Int_t cls ) const;
      void P2X( std::vector<Float_t>&, const std::vector<Float_t>&, Int_t cls ) const;

      // can't say if we want to use them
      // store relevant parts of PCA locally
      std::vector<TVectorD*> fMeanValues;   // mean values
      std::vector<TMatrixD*> fEigenVectors; // eigenvectors

      ClassDef(VariableAutoEncoderTransform,0); // Variable transformation: Principal Value Composition
   };

} // namespace TMVA


#endif /* ROOT_TMVA_VARIABLE_AUTOENCODER_HANDLER */
