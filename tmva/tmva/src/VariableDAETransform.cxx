// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Eckhard von Toerne

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableDAETransform                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>     - U of Bonn, Germany          *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Marc Huwiler    <marc.huwiler@windowslive.com> - CERN, Switzerland        *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::VariableDAETransform
\ingroup TMVA
Linear interpolation class
*/

#include "TMVA/VariableDAETransform.h"

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
#include "TMatrix.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"


#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

ClassImp(TMVA::VariableDAETransform);

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::VariableDAETransform::VariableDAETransform( DataSetInfo& dsi )
: VariableTransformBase( dsi, Types::kDAETransform, "DAETransform" )
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::VariableDAETransform::~VariableDAETransform()
{
   for (UInt_t i=0; i<fMeanValues.size(); i++) {
      if (fMeanValues.at(i)   != 0) delete fMeanValues.at(i);
      if (fEigenVectors.at(i) != 0) delete fEigenVectors.at(i);
   }
   for (size_t i = 0; i<fAutoEncoder.size(); i++) 
   {
      delete fAutoEncoder[i]; 
   }
}

////////////////////////////////////////////////////////////////////////////////
/// initialization of the transformation.
/// Has to be called in the preparation and not in the constructor,
/// since the number of classes it not known at construction, but
/// only after the creation of the DataSet which might be later.

//template <typename Architecture_t>
void TMVA::VariableDAETransform::Initialize()
{
   
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the principal components using the ROOT class TPrincipal
/// and the normalization

Bool_t TMVA::VariableDAETransform::PrepareTransformation (const std::vector<Event*>& events)
{
   
   Initialize();


   if (!IsEnabled() || IsCreated()) return kTRUE;

   Log() << kINFO << "Preparing the Deep Autoencoder transformation..." << Endl;

   UInt_t inputSize = fGet.size();

   SetNVariables(inputSize);

   // TPrincipal doesn't support PCA transformation for 1 or less variables
   if (inputSize <= 1) {
      Log() << kFATAL << "Cannot perform DAETransform for " << inputSize << " variable only" << Endl;
      return kFALSE;
   }

   if (inputSize > 200) {
      Log() << kINFO << "----------------------------------------------------------------------------"
            << Endl;
      Log() << kINFO
            << ": More than 200 variables, will not calculate DAETransform!" << Endl;
      Log() << kINFO << "----------------------------------------------------------------------------"
            << Endl;
      return kFALSE;
   }

   TrainOnExampleData( events );

   std::cout << "PrepareTransfiormation succeded " << std::endl; 

   SetCreated( kTRUE );

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// apply the principal component analysis

const TMVA::Event* TMVA::VariableDAETransform::Transform( const Event* const ev, Int_t cls ) const
{
   if (!IsCreated()) return 0;

   //   const Int_t inputSize = fGet.size();
   //   const UInt_t nCls = GetNClasses();

   // if we have more than one class, take the last PCA analysis where all classes are combined if
   // the cls parameter is outside the defined classes
   // If there is only one class, then no extra class for all events of all classes has to be created

   //if (cls < 0 || cls > GetNClasses()) cls = (fMeanValues.size()==1?0:2);//( GetNClasses() == 1 ? 0 : 1 );  ;
   // EVT this is a workaround to address the reader problem with transforma and EvaluateMVA(std::vector<float/double> ,...)
   if (cls < 0 || cls >= (int) fMeanValues.size()) cls = fMeanValues.size()-1;
   // EVT workaround end

   // Perform PCA and put it into PCAed events tree

   if (fTransformedEvent==0 ) {
      fTransformedEvent = new Event();
   }

   std::vector<Float_t> localInput;
   std::vector<Char_t>  mask;
   std::vector<Float_t> principalComponents;

   Bool_t hasMaskedEntries = GetInput( ev, localInput, mask );

   if( hasMaskedEntries ){ // targets might be masked (for events where the targets have not been computed yet)
      UInt_t numMasked = std::count(mask.begin(), mask.end(), (Char_t)kTRUE);
      UInt_t numOK     = std::count(mask.begin(), mask.end(), (Char_t)kFALSE);
      if( numMasked>0 && numOK>0 ){
         Log() << kFATAL << "You mixed variables and targets in the Deep Autoencoder transformation. This is not possible." << Endl;
      }
      SetOutput( fTransformedEvent, localInput, mask, ev );
      return fTransformedEvent;
   }

   Matrix_t transformedEvent; 
   std::vector<Matrix_t> transformedEvents; 
   transformedEvents.push_back(transformedEvent); 

   //BackTransformOutputData(transformedEvent, localInput); 
   TransformInputData(localInput, transformedEvents[0]); 

   for (unsigned int i=0; i<fAutoEncoder.size(); i++) 
   {
      fAutoEncoder[i]->FineTune(transformedEvents, transformedEvents, transformedEvents, 2, 1, 0.1, 10); 
   }

   //X2P( principalComponents, localInput, cls );
   SetOutput( fTransformedEvent, localInput, mask, ev ); 

   return fTransformedEvent;
}

/*void TMVA::VariableDAETransform::CreateOutput( Event* event, std::vector<Matrix_t>& output, std::vector<Char_t>& mask, const Event* oldEvent, Bool_t backTransformation) const
{
   std::vector<Float_t>::iterator itOutput = output.begin();
   std::vector<Char_t>::iterator  itMask   = mask.begin();

   if( oldEvent )
      event->CopyVarValues( *oldEvent );

   try {

      ItVarTypeIdxConst itEntry;
      ItVarTypeIdxConst itEntryEnd;

      if( backTransformation || fPut.empty() ){ // as in GetInput, but the other way round (from fPut for transformation, from fGet for backTransformation)
         itEntry = fGet.begin();
         itEntryEnd = fGet.end();
      }
      else {
         itEntry = fPut.begin();
         itEntryEnd = fPut.end();
      }


      for( ; itEntry != itEntryEnd; ++itEntry ) {

         if( (*itMask) ){ // if the value is masked
            continue;
         }

         Char_t type = (*itEntry).first;
         Int_t  idx  = (*itEntry).second;
         if (itOutput == output.end()) Log() << kFATAL << "Read beyond array boundaries in VariableTransformBase::SetOutput"<<Endl;
         Float_t value = (*itOutput);

         switch( type ) {
         case 'v':
            event->SetVal( idx, value );
            break;
         case 't':
            event->SetTarget( idx, value );
            break;
         case 's':
            event->SetSpectator( idx, value );
            break;
         default:
            Log() << kFATAL << "VariableTransformBase/GetInput : unknown type '" << type << "'." << Endl;
         }
         if( !(*itMask) ) ++itOutput;
         ++itMask;

      }
   }catch( std::exception& except ){
      Log() << kFATAL << "VariableTransformBase/SetOutput : exception/" << except.what() << Endl;
      throw;
   }
}*/

////////////////////////////////////////////////////////////////////////////////
/// apply the principal component analysis
/// TODO: implementation of inverse transformation
///    Log() << kFATAL << "Inverse transformation for PCA transformation not yet implemented. Hence, this transformation cannot be applied together with regression. Please contact the authors if necessary." << Endl;

const TMVA::Event* TMVA::VariableDAETransform::InverseTransform( const Event* const ev, Int_t cls ) const
{
   if (!IsCreated()) return 0;
   //   const Int_t inputSize = fGet.size();
   const UInt_t nCls = GetNClasses();
   //UInt_t evCls = ev->GetClass();

   // if we have more than one class, take the last PCA analysis where all classes are combined if
   // the cls parameter is outside the defined classes
   // If there is only one class, then no extra class for all events of all classes has to be created
   if (cls < 0 || UInt_t(cls) > nCls) cls = (fMeanValues.size()==1?0:2);//( GetNClasses() == 1 ? 0 : 1 );  ;
   // Perform PCA and put it into PCAed events tree

   if (fBackTransformedEvent==0 ) fBackTransformedEvent = new Event();

   std::vector<Float_t> principalComponents;
   std::vector<Char_t>  mask;
   std::vector<Float_t> output;

   GetInput( ev, principalComponents, mask, kTRUE );
   P2X( output, principalComponents, cls );
   SetOutput( fBackTransformedEvent, output, mask, ev, kTRUE );

   return fBackTransformedEvent;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the principal components for the signal and the background data
/// it uses the MakePrincipal method of ROOT's TPrincipal class

void TMVA::VariableDAETransform::TrainOnExampleData( const std::vector< Event*>& events )
{
   size_t BatchSize = 1;
   size_t InputDepth = 1;     // Just put 1 here
   size_t InputHeight = 1; 
   size_t InputWidth = 1; 
   size_t BatchDepth = 1; 
   size_t BatchHeight = 1; 
   size_t BatchWidth = 1; 
   DNN::ELossFunction fJ = DNN::ELossFunction::kCrossEntropy; 
   DNN::EInitialization fI = DNN::EInitialization::kGauss; 
   DNN::ERegularization fR = DNN::ERegularization::kNone; 
   Scalar_t fWeightDecay = 0.0; 
   bool isTraining = false; 

   std::vector<size_t> numHiddenUnitsPerLayer; 
   Scalar_t learningRate = 0.1; 
   Scalar_t corruptionLevel = 0.3; 
   Scalar_t dropoutProbability = 1.; 
   size_t epochs = 50; 
   DNN::EActivationFunction activation; 
   bool applyDropout = false; 

   numHiddenUnitsPerLayer.push_back(2); 
   activation = DNN::EActivationFunction::kSoftSign; 


   UInt_t nvars = 0, ntgts = 0, nspcts = 0;
   CountVariableTypes( nvars, ntgts, nspcts );
   if( nvars>0  && ntgts>0 )
      Log() << kFATAL << "Variables and targets cannot be mixed in DeepAutoEncoder transformation." << Endl;



   const Int_t inputSize = fGet.size();

   // if we have more than one class, add another PCA analysis which combines all classes
   const UInt_t nCls = GetNClasses();
   const UInt_t numDAE = (nCls<=1) ? nCls : nCls+1;
// ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

   // PCA [signal/background/class x/class y/... /all classes]
   //std::vector<DNN::TDeepAutoEncoder<Architecture_t>* > DAE(numDAE);
   for (UInt_t i=0; i<numDAE; i++) fAutoEncoder.push_back( new TMVA::DNN::TDeepAutoEncoder<Architecture_t>(BatchSize, InputDepth, InputHeight, InputWidth, 
                                       BatchDepth, BatchHeight, BatchWidth, fJ, fI, fR, fWeightDecay, isTraining) ); 

   // !! Not normalizing and not storing input data, for performance reasons. Should perhaps restore normalization.
   // But this can be done afterwards by adding a normalisation transformation (user defined)

   //std::vector<Matrix_t> inputs; // We already have this in the class. 

   size_t visibleUnits = events[0]->GetValues().size(); 
   size_t numEvents = events.size(); 

   BatchSize = numEvents; 

   //TransformInputDataset(events, input); 

   size_t hiddenUnits = 2; 

   
   std::vector<Float_t> bareinput;
   std::vector<Char_t>  mask;

   input.clear(); 

   for ( unsigned int i = 0; i<numEvents; i++ ) 
   {
      input.emplace_back(visibleUnits, 1);
      
      const Event* ev = events[i];        // Why this? Can't we just pass events[i] in the function?
      UInt_t cls = ev->GetClass();

      Bool_t hasMaskedEntries = GetInput( ev, bareinput, mask );

      if (hasMaskedEntries){
         Log() << kWARNING << "Print event which triggers an error" << Endl;
         std::ostringstream oss;
         ev->Print(oss);
         Log() << oss.str();
         Log() << kFATAL << "Masked entries found in event read in when calculating the principal components for the PCA transformation." << Endl;
      }

      /*UInt_t iinp = 0;
      for( std::vector<Float_t>::iterator itInp = bareinput.begin(), itInpEnd = bareinput.end(); itInp != itInpEnd; ++itInp )
         {
            Float_t value = (*itInp);
            dvec[iinp] = (Double_t)value;
            ++iinp;
         }*/

      //DAE.at(cls)->AddRow( dvec );
      //if (nCls > 1) DAE.at(numDAE-1)->AddRow( dvec );
      TransformInputData(bareinput, input[i]); 
   }
   for (unsigned int i=0; i<input.size(); i++) 
   {
      for (int j=0; j<input[i].GetNrows(); j++) 
      {
         std::cout << input[i](j, 0) << " "; 
      }
      std::cout << std::endl; 
   }
   
   
   // delete possible leftovers
   for (UInt_t i=0; i<fMeanValues.size(); i++)   if (fMeanValues[i]   != 0) delete fMeanValues[i];
   for (UInt_t i=0; i<fEigenVectors.size(); i++) if (fEigenVectors[i] != 0) delete fEigenVectors[i];
   fMeanValues.resize(numDAE,0);
   fEigenVectors.resize(numDAE,0);

   for (UInt_t i=0; i<numDAE; i++ ) {
      std::cout << "Training autoencoder " << i << std::endl; 
      fAutoEncoder.at(i)->PreTrain(input, numHiddenUnitsPerLayer, learningRate, corruptionLevel, dropoutProbability, epochs, activation, applyDropout); 
   }

   std::cout << std::endl << "Training successful! " << std::endl; 

   //for (UInt_t i=0; i<numDAE; i++) delete DAE.at(i);
   //delete [] dvec;
}

/*Bool_t TMVA::VariableDAETransform::GetEventValues( const Event* event, Matrix_t& input, std::vector<Char_t>& mask, Bool_t backTransformation ) const
{
   ItVarTypeIdxConst itEntry;
   ItVarTypeIdxConst itEntryEnd;

   //input.clear();
   mask.clear();

   if( backTransformation && !fPut.empty() ){
      itEntry = fPut.begin();
      itEntryEnd = fPut.end();
      //input.reserve(fPut.size());
   }
   else {
      itEntry = fGet.begin();
      itEntryEnd = fGet.end();
      //input.reserve(fGet.size() );
   }

   Bool_t hasMaskedEntries = kFALSE;
   //   event->Print(std::cout);
   for( ; itEntry != itEntryEnd; ++itEntry ) {
      Char_t type = (*itEntry).first;
      size_t idx  = (*itEntry).second;
      
      input = Matrix_t( event->GetValues().size() , 1);

      try{
         switch( type ) {
         case 'v':
            input(idx, 0) = event->GetValue(idx); 
            break;
         case 't':
            input(idx, 0) = event->GetTarget(idx); 
            break;
         case 's':
            input(idx, 0) = event->GetSpectator(idx); 
            break;
         default:
            Log() << kFATAL << "VariableTransformBase/GetInput : unknown type '" << type << "'." << Endl;
         }
         mask.push_back(kFALSE);
      }
      catch(std::out_of_range& /* excpt * ){ // happens when an event is transformed which does not yet have the targets calculated (in the application phase)
         input(idx, 0) = 0.f; 
         mask.push_back(kTRUE);
         hasMaskedEntries = kTRUE;
      }
   }
   return hasMaskedEntries;
}*/

void TMVA::VariableDAETransform::TransformInputDataset( const std::vector< Event*>& localEvents, std::vector<Matrix_t>& localInputs) 
{
   size_t visibleUnits = localEvents[0]->GetValues().size(); 
   size_t numEvents = localEvents.size(); 
   for ( unsigned int i = 0; i<numEvents; i++ ) 
   {
      input.emplace_back(visibleUnits, 1); 
      for (unsigned int j = 0; j < visibleUnits; j++) 
      {
         localInputs[i](j, 0) = localEvents[i]->GetValues()[j]; 
      }
   }
}

void TMVA::VariableDAETransform::TransformInputData( const std::vector<Float_t>& localEvent, Matrix_t& remoteInput) const 
{
   //std::cout << "Starting conversion from vector<Float_t> to Matrix_t " << std::endl; 
   size_t numVar = localEvent.size(); 
   Matrix_t localInput(numVar, 1); 
   for ( unsigned int i = 0; i<2/*numVar*/; i++ ) 
   {
      localInput(i, 0) = localEvent[i]; 
   }
   remoteInput = localInput; 
}

void TMVA::VariableDAETransform::BackTransformOutputData( const Matrix_t& autoencoderOutput, std::vector<Float_t>& vec) const 
{
   /*for (unsigned int i=0; i<numEvents; i++) 
   {
      output.emplace_back(hiddenUnits, 1); 
   }*/
   std::vector<Float_t> outputVector; 
   if (autoencoderOutput.GetNcols()<2) 
   {
      for (unsigned int i=0; i< autoencoderOutput.GetNrows(); i++)
         outputVector.push_back(autoencoderOutput(i, 0)); 
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the principal components from the original data vector
/// x, and return it in p (function extracted from TPrincipal::X2P)
/// It's the users responsibility to make sure that both x and p are
/// of the right size (i.e., memory must be allocated for p)

void TMVA::VariableDAETransform::X2P( std::vector<Float_t>& pc, const std::vector<Float_t>& x, Int_t cls ) const
{
   const Int_t nInput = x.size();
   pc.assign(nInput,0);

   for (Int_t i = 0; i < nInput; i++) {
      Double_t pv = 0;
      for (Int_t j = 0; j < nInput; j++)
         pv += (((Double_t)x.at(j)) - (*fMeanValues.at(cls))(j)) * (*fEigenVectors.at(cls))(j,i);
      pc[i] = pv;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Perform the back-transformation from the principal components
/// pc, and return x
/// It's the users responsibility to make sure that both x and pc are
/// of the right size (i.e., memory must be allocated for p)

void TMVA::VariableDAETransform::P2X( std::vector<Float_t>& x, const std::vector<Float_t>& pc, Int_t cls ) const
{
   const Int_t nInput = pc.size();
   x.assign(nInput,0);

   for (Int_t i = 0; i < nInput; i++) {
      Double_t xv = 0;
      for (Int_t j = 0; j < nInput; j++)
         xv += (((Double_t)pc.at(j)) * (*fEigenVectors.at(cls))(i,j) ) + (*fMeanValues.at(cls))(j);
      x[i] = xv;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// write mean values to stream

void TMVA::VariableDAETransform::WriteTransformationToStream( std::ostream& o ) const
{
   for (Int_t sbType=0; sbType<2; sbType++) {
      o << "# PCA mean values " << std::endl;
      const TVectorD* means = fMeanValues[sbType];
      o << (sbType==0 ? "Signal" : "Background") << " " << means->GetNrows() << std::endl;
      for (Int_t row = 0; row<means->GetNrows(); row++) {
         o << std::setprecision(12) << std::setw(20) << (*means)[row];
      }
      o << std::endl;
   }
   o << "##" << std::endl;

   // write eigenvectors to stream
   for (Int_t sbType=0; sbType<2; sbType++) {
      o << "# PCA eigenvectors " << std::endl;
      const TMatrixD* mat = fEigenVectors[sbType];
      o << (sbType==0 ? "Signal" : "Background") << " " << mat->GetNrows() << " x " << mat->GetNcols() << std::endl;
      for (Int_t row = 0; row<mat->GetNrows(); row++) {
         for (Int_t col = 0; col<mat->GetNcols(); col++) {
            o << std::setprecision(12) << std::setw(20) << (*mat)[row][col] << " ";
         }
         o << std::endl;
      }
   }
   o << "##" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// create XML description of PCA transformation

void TMVA::VariableDAETransform::AttachXMLTo(void* parent) {
   void* trfxml = gTools().AddChild(parent, "Transform");
   gTools().AddAttr(trfxml, "Name", "PCA");

   VariableTransformBase::AttachXMLTo( trfxml );

   // write mean values to stream
   for (UInt_t sbType=0; sbType<fMeanValues.size(); sbType++) {
      void* meanxml = gTools().AddChild( trfxml, "Statistics");
      const TVectorD* means = fMeanValues[sbType];
      gTools().AddAttr( meanxml, "Class",     (sbType==0 ? "Signal" :(sbType==1 ? "Background":"Combined")) );
      gTools().AddAttr( meanxml, "ClassIndex", sbType );
      gTools().AddAttr( meanxml, "NRows",      means->GetNrows() );
      TString meansdef = "";
      for (Int_t row = 0; row<means->GetNrows(); row++)
         meansdef += gTools().StringFromDouble((*means)[row]) + " ";
      gTools().AddRawLine( meanxml, meansdef );
   }

   // write eigenvectors to stream
   for (UInt_t sbType=0; sbType<fEigenVectors.size(); sbType++) {
      void* evxml = gTools().AddChild( trfxml, "Eigenvectors");
      const TMatrixD* mat = fEigenVectors[sbType];
      gTools().AddAttr( evxml, "Class",      (sbType==0 ? "Signal" :(sbType==1 ? "Background":"Combined") ) );
      gTools().AddAttr( evxml, "ClassIndex", sbType );
      gTools().AddAttr( evxml, "NRows",      mat->GetNrows() );
      gTools().AddAttr( evxml, "NCols",      mat->GetNcols() );
      TString evdef = "";
      for (Int_t row = 0; row<mat->GetNrows(); row++)
         for (Int_t col = 0; col<mat->GetNcols(); col++)
            evdef += gTools().StringFromDouble((*mat)[row][col]) + " ";
      gTools().AddRawLine( evxml, evdef );
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Read the transformation matrices from the xml node

void TMVA::VariableDAETransform::ReadFromXML( void* trfnode )
{
   Int_t nrows, ncols;
   UInt_t clsIdx;
   TString classtype;
   TString nodeName;

   Bool_t newFormat = kFALSE;

   void* inpnode = NULL;

   inpnode = gTools().GetChild(trfnode, "Selection"); // new xml format
   if( inpnode!=NULL )
      newFormat = kTRUE; // new xml format

   if( newFormat ){
      // ------------- new format --------------------
      // read input
      VariableTransformBase::ReadFromXML( inpnode );

   }

   void* ch = gTools().GetChild(trfnode);
   while (ch) {
      nodeName = gTools().GetName(ch);
      if (nodeName == "Statistics") {
         // read mean values
         gTools().ReadAttr(ch, "Class",      classtype);
         gTools().ReadAttr(ch, "ClassIndex", clsIdx);
         gTools().ReadAttr(ch, "NRows",      nrows);

         // set the correct size
         if (fMeanValues.size()<=clsIdx) fMeanValues.resize(clsIdx+1,0);
         if (fMeanValues[clsIdx]==0) fMeanValues[clsIdx] = new TVectorD( nrows );
         fMeanValues[clsIdx]->ResizeTo( nrows );

         // now read vector entries
         std::stringstream s(gTools().GetContent(ch));
         for (Int_t row = 0; row<nrows; row++) s >> (*fMeanValues[clsIdx])(row);
      }
      else if ( nodeName == "Eigenvectors" ) {
         // Read eigenvectors
         gTools().ReadAttr(ch, "Class",      classtype);
         gTools().ReadAttr(ch, "ClassIndex", clsIdx);
         gTools().ReadAttr(ch, "NRows",      nrows);
         gTools().ReadAttr(ch, "NCols",      ncols);

         if (fEigenVectors.size()<=clsIdx) fEigenVectors.resize(clsIdx+1,0);
         if (fEigenVectors[clsIdx]==0) fEigenVectors[clsIdx] = new TMatrixD( nrows, ncols );
         fEigenVectors[clsIdx]->ResizeTo( nrows, ncols );

         // now read matrix entries
         std::stringstream s(gTools().GetContent(ch));
         for (Int_t row = 0; row<nrows; row++)
            for (Int_t col = 0; col<ncols; col++)
               s >> (*fEigenVectors[clsIdx])[row][col];
      } // done reading eigenvectors
      ch = gTools().GetNextChild(ch);
   }

   SetCreated();
}

////////////////////////////////////////////////////////////////////////////////
/// Read mean values from input stream

void TMVA::VariableDAETransform::ReadTransformationFromStream( std::istream& istr, const TString& classname )
{
   char buf[512];
   istr.getline(buf,512);
   TString strvar, dummy;
   Int_t nrows(0), ncols(0);
   UInt_t classIdx=(classname=="signal"?0:1);

   for (UInt_t i=0; i<fMeanValues.size(); i++) {
      if (fMeanValues.at(i)   != 0) delete fMeanValues.at(i);
      if (fEigenVectors.at(i) != 0) delete fEigenVectors.at(i);
   }
   fMeanValues.resize(3);
   fEigenVectors.resize(3);

   Log() << kINFO << "VariableDAETransform::ReadTransformationFromStream(): " << Endl;

   while (!(buf[0]=='#'&& buf[1]=='#')) { // if line starts with ## return
      char* p = buf;
      while (*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512);
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> strvar;
      if (strvar=="signal" || strvar=="background") {

         sstr >> nrows;
         Int_t sbType = (strvar=="signal" ? 0 : 1);

         if (fMeanValues[sbType] == 0) fMeanValues[sbType] = new TVectorD( nrows );
         else                          fMeanValues[sbType]->ResizeTo( nrows );

         // now read vector entries
         for (Int_t row = 0; row<nrows; row++) istr >> (*fMeanValues[sbType])(row);

      } // done reading vector

      istr.getline(buf,512); // reading the next line
   }

   // Read eigenvectors from input stream
   istr.getline(buf,512);
   while (!(buf[0]=='#'&& buf[1]=='#')) { // if line starts with ## return
      char* p = buf;
      while(*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512);
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> strvar;
      if (strvar=="signal" || strvar=="background") {

         // coverity[tainted_data_argument]
         sstr >> nrows >> dummy >> ncols;
         Int_t sbType = (strvar=="signal" ? 0 : 1);

         if (fEigenVectors[sbType] == 0) fEigenVectors[sbType] = new TMatrixD( nrows, ncols );
         else                            fEigenVectors[sbType]->ResizeTo( nrows, ncols );

         // now read matrix entries
         for (Int_t row = 0; row<fEigenVectors[sbType]->GetNrows(); row++) {
            for (Int_t col = 0; col<fEigenVectors[sbType]->GetNcols(); col++) {
               istr >> (*fEigenVectors[sbType])[row][col];
            }
         }

      } // done reading matrix
      istr.getline(buf,512); // reading the next line
   }
   fMeanValues[2] = new TVectorD( *fMeanValues[classIdx] );
   fEigenVectors[2] = new TMatrixD( *fEigenVectors[classIdx] );

   SetCreated();
}

////////////////////////////////////////////////////////////////////////////////
/// creates C++ code fragment of the PCA transform for inclusion in standalone C++ class

void TMVA::VariableDAETransform::MakeFunction( std::ostream& fout, const TString& fcncName,
                                               Int_t part, UInt_t trCounter, Int_t )
{
   UInt_t nvar = fEigenVectors[0]->GetNrows();

   // creates a PCA transformation function
   UInt_t numC = fMeanValues.size();
   if (part==1) {
      fout << std::endl;
      fout << "   void X2P_"<<trCounter<<"( const double*, double*, int ) const;" << std::endl;
      fout << "   double fMeanValues_"<<trCounter<<"["<<numC<<"]["
           << fMeanValues[0]->GetNrows()   << "];" << std::endl;   // mean values
      fout << "   double fEigenVectors_"<<trCounter<<"["<<numC<<"]["
           << fEigenVectors[0]->GetNrows() << "]["
           << fEigenVectors[0]->GetNcols() <<"];" << std::endl;   // eigenvectors
      fout << std::endl;
   }

   // sanity check
   if (numC>1){
      if (fMeanValues[0]->GetNrows()   != fMeanValues[1]->GetNrows() ||
          fEigenVectors[0]->GetNrows() != fEigenVectors[1]->GetNrows() ||
          fEigenVectors[0]->GetNcols() != fEigenVectors[1]->GetNcols()) {
         Log() << kFATAL << "<MakeFunction> Mismatch in vector/matrix dimensions" << Endl;
      }
   }

   if (part==2) {

      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::X2P_"<<trCounter<<"( const double* x, double* p, int index ) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   // Calculate the principal components from the original data vector" << std::endl;
      fout << "   // x, and return it in p (function extracted from TPrincipal::X2P)" << std::endl;
      fout << "   // It's the users responsibility to make sure that both x and p are" << std::endl;
      fout << "   // of the right size (i.e., memory must be allocated for p)." << std::endl;
      fout << "   const int nVar = " << nvar << ";" << std::endl;
      fout << std::endl;
      fout << "   for (int i = 0; i < nVar; i++) {" << std::endl;
      fout << "      p[i] = 0;" << std::endl;
      fout << "      for (int j = 0; j < nVar; j++) p[i] += (x[j] - fMeanValues_"<<trCounter<<"[index][j]) * fEigenVectors_"<<trCounter<<"[index][j][i];" << std::endl;
      fout << "   }" << std::endl;
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::InitTransform_"<<trCounter<<"()" << std::endl;
      fout << "{" << std::endl;
      fout << "   // PCA transformation, initialisation" << std::endl;

      // fill vector of mean values
      fout << "   // initialise vector of mean values" << std::endl;
      std::streamsize dp = fout.precision();
      for (UInt_t index=0; index<numC; index++) {
         for (int i=0; i<fMeanValues[index]->GetNrows(); i++) {
            fout << "   fMeanValues_"<<trCounter<<"["<<index<<"]["<<i<<"] = " << std::setprecision(12)
                 << (*fMeanValues[index])(i) << ";" << std::endl;
         }
      }

      // fill matrix of eigenvectors
      fout << std::endl;
      fout << "   // initialise matrix of eigenvectors" << std::endl;
      for (UInt_t index=0; index<numC; index++) {
         for (int i=0; i<fEigenVectors[index]->GetNrows(); i++) {
            for (int j=0; j<fEigenVectors[index]->GetNcols(); j++) {
               fout << "   fEigenVectors_"<<trCounter<<"["<<index<<"]["<<i<<"]["<<j<<"] = " << std::setprecision(12)
                    << (*fEigenVectors[index])(i,j) << ";" << std::endl;
            }
         }
      }
      fout << std::setprecision(dp);
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::Transform_"<<trCounter<<"( std::vector<double>& iv, int cls ) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   // PCA transformation" << std::endl;
      fout << "   const int nVar = " << nvar << ";" << std::endl;
      fout << "   double *dv = new double[nVar];" << std::endl;
      fout << "   double *rv = new double[nVar];" << std::endl;
      fout << "   if (cls < 0 || cls > "<<GetNClasses()<<") {"<< std::endl;
      fout << "       if ("<<GetNClasses()<<" > 1 ) cls = "<<GetNClasses()<<";"<< std::endl;
      fout << "       else cls = "<<(numC==1?0:2)<<";"<< std::endl;
      fout << "   }"<< std::endl;

      VariableTransformBase::MakeFunction(fout, fcncName, 0, trCounter, 0 );

      fout << "   for (int ivar=0; ivar<nVar; ivar++) dv[ivar] = iv[indicesGet.at(ivar)];" << std::endl;

      fout << std::endl;
      fout << "   // Perform PCA and put it into PCAed events tree" << std::endl;
      fout << "   this->X2P_"<<trCounter<<"( dv, rv, cls );" << std::endl;
      fout << "   for (int ivar=0; ivar<nVar; ivar++) iv[indicesPut.at(ivar)] = rv[ivar];" << std::endl;

      fout << std::endl;
      fout << "   delete [] dv;" << std::endl;
      fout << "   delete [] rv;" << std::endl;
      fout << "}" << std::endl;
   }
}