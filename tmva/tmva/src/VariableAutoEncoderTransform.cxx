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


#include "TMVA/VariableAutoEncoderTransform.h"

#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"

#include "TMatrixD.h"
#include "TMatrixDBase.h"
#include "TPrincipal.h"
#include "TVectorD.h"
#include "TVectorF.h"

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
/*
ClassImp(TMVA::VariableAutoEncoderTransform);

 ////////////////////////////////////////////////////////////////////////////////
 /// constructor

TMVA::VariableAutoEncoderTransform::VariableAutoEncoderTransform( DataSetInfo& dsi )
   : VariableTransformBase( dsi, Types::kAUTOENCODER, "AUTOENCODER" )
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::VariableAutoEncoderTransform::~VariableAutoEncoderTransform()
{
  //don't know if this is to be used
   //for (UInt_t i=0; i<fMeanValues.size(); i++) {
    //  if (fMeanValues.at(i)   != 0) delete fMeanValues.at(i);
      //if (fEigenVectors.at(i) != 0) delete fEigenVectors.at(i);
   //}
}

////////////////////////////////////////////////////////////////////////////////
/// initialization of the transformation.
/// Has to be called in the preparation and not in the constructor,
/// since the number of classes it not known at construction, but
/// only after the creation of the DataSet which might be later.

void TMVA::VariableAutoEncoderTransform::Initialize()
{
}

//______________________________________________________________________________
//Bool_t TMVA::VariableAutoEncoderTransform::PrepareTransformation (const std::vector<Event*>& events)
//{

//}

//______________________________________________________________________________

//const TMVA::Event* TMVA::VariableAutoEncoderTransform::Transform( const Event* const ev, Int_t cls ) const
//{

//}

//______________________________________________________________________________

//const TMVA::Event* TMVA::VariableAutoEncoderTransform::InverseTransform( const Event* const ev, Int_t cls ) const
//{
//}

//______________________________________________________________________________

void TMVA::VariableAutoEncoderTransform::CalculatePrincipalComponents( const std::vector< Event*>& events )
{

}
//______________________________________________________________________________

//void TMVA::VariableAutoEncoderTransform::CalculatePrincipalComponents( const std::vector< Event*>& events )
{
}
//______________________________________________________________________________

void TMVA::VariableAutoEncoderTransform::P2X( std::vector<Float_t>& x, const std::vector<Float_t>& pc, Int_t cls ) const
{

}
//______________________________________________________________________________
void TMVA::VariableAutoEncoderTransform::WriteTransformationToStream( std::ostream& o ) const
{
}
//______________________________________________________________________________

void TMVA::VariableAutoEncoderTransform::AttachXMLTo(void* parent) {
}
//______________________________________________________________________________
void TMVA::VariableAutoEncoderTransform::ReadFromXML( void* trfnode )
{}

//______________________________________________________________________________

////////////////////////////////////////////////////////////////////////////////
/// Read mean values from input stream

void TMVA::VariableAutoEncoderTransform::ReadTransformationFromStream( std::istream& istr, const TString& classname )
{}

//______________________________________________________________________________


////////////////////////////////////////////////////////////////////////////////
/// creates C++ code fragment of the PCA transform for inclusion in standalone C++ class

void TMVA::VariableAutoEncoderTransform::MakeFunction( std::ostream& fout, const TString& fcncName,
                                               Int_t part, UInt_t trCounter, Int_t )
{}
*/
//______________________________________________________________________________
