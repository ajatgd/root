// @(#)root/tmva $Id$
// Author: Marc Huwiler

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
 *      Marc Huwiler    <marc.huwiler@windowslive.com> - CERN, Switzerland        *  
 *      Akshay Vashistha <akshayvashistha1995@gmail.com>  - JSSATE, Noida, India  *
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
: VariableTransformBase( dsi, Types::kDAETransform, "DAETransform" ),
   numCompressedUnits(0)
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


Bool_t TMVA::VariableDAETransform::PrepareTransformation (const std::vector<Event*>& events)
{

   Initialize();


   if (!IsEnabled() || IsCreated()) return kTRUE;

   Log() << kINFO << "Preparing the Deep Autoencoder transformation..." << Endl;

   UInt_t inputSize = fGet.size();

   SetNVariables(inputSize);

   // 
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

   std::cout << "PrepareTransformation succeded " << std::endl;

   SetCreated( kTRUE );

   std::ofstream file; 
   file.open("/home/giuseppe/rootauto/testing/outpufuncDAE.cxx"); 
   MakeFunction(file, "function", 2, 2, 2); 
   file.close(); 

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// apply the DAE Transformation

const TMVA::Event* TMVA::VariableDAETransform::Transform( const Event* const ev, Int_t cls ) const
{
   if (!IsCreated()) return 0;

   // if we have more than one class, take the last DAE analysis where all classes are combined if
   // the cls parameter is outside the defined classes
   // If there is only one class, then no extra class for all events of all classes has to be created

   //if (cls < 0 || cls > GetNClasses()) cls = (fMeanValues.size()==1?0:2);//( GetNClasses() == 1 ? 0 : 1 );  ;
   // EVT this is a workaround to address the reader problem with transforma and EvaluateMVA(std::vector<float/double> ,...)
   if (cls < 0 || cls >= (int) input.size()) cls = input.size()-1;
   // EVT workaround end

   // Perform DAETransform and return pointers to the new events.

   if (fTransformedEvent==0 ) {
      fTransformedEvent = new Event();
   }
   Int_t currentClass = ev->GetClass();

   Matrix_t transformedEvent, encodedEvent;
   std::vector<Float_t> localInput, localOutput;
   std::vector<Char_t>  mask;
   encodedEvent.ResizeTo(numCompressedUnits, 1);

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


   TransformInputData(localInput, transformedEvent);
   
   encodedEvent = fAutoEncoder[currentClass]->PredictEncodedOutput(transformedEvent);      // Maybe cls for same behaviour as PCA...

   BackTransformOutputData(encodedEvent, localOutput);
   
   SetOutput( fTransformedEvent, localOutput, mask, ev );


   //SetOutputDataSetInfo(new DataSetInfo((*fDsi)));    // No copy default method... 

   

   //std::cout << "fTransformedEvent : " << fTransformedEvent->GetNVariables() << std::endl; 
   if (true) {
   const Event* decodedEvent = InverseTransform(fTransformedEvent, 2);

   std::ofstream myfile, differencefile;
   myfile.open("/home/giuseppe/rootauto/testing/autoencodertest.txt", std::ios::app);
   differencefile.open("/home/giuseppe/rootauto/testing/differences.txt", std::ios::app);
   for (unsigned int i=0; i<fTransformedEvent->GetNVariables(); i++)
   {
      myfile << fTransformedEvent->GetValue(i) << " ";

   }
   myfile << " ";
   //assert(decodedEvent->GetNVariables() == ev->GetNVariables()); 
   for (unsigned int i=0; i<decodedEvent->GetNVariables(); i++)
   {
      myfile << decodedEvent->GetValue(i) << " ";
      differencefile << (ev->GetValue(i) - decodedEvent->GetValue(i)) << " "; 
   }
   myfile << std::endl; 
   differencefile << std::endl; 
   myfile.close();
   differencefile.close(); 

   }

   return fTransformedEvent;
}

/*void TMVA::VariableDAETransform::CreateOutput( Event* event, std::vector<Matrix_t>& output, std::vector<Char_t>& mask, const Event* oldEvent, Bool_t backTransformation) const
{
   std::vector<Float_t>::iterator itOutput = output.begin();
   std::vector<Char_t>::iterator  itMask   = mask.begin();

   if( oldEvent )
      event->CopyVarValues( *oldEvent );

   event->ResizeValues(output.size());


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
///    Log() << kFATAL << "Inverse transformation for DAE transformation not yet implemented. Hence, this transformation cannot be applied together with regression. Please contact the authors if necessary." << Endl;

const TMVA::Event* TMVA::VariableDAETransform::InverseTransform( const Event* const ev, Int_t cls ) const
{
   if (!IsCreated()) return 0;
   //   const Int_t inputSize = fGet.size();
   //const UInt_t nCls = GetNClasses();
   Int_t currentClass = ev->GetClass();

   if (fBackTransformedEvent==0 ) fBackTransformedEvent = new Event();

   Matrix_t backTransformInput, backTransformOutput;
   std::vector<Float_t> localInput, localOutput;
   std::vector<Char_t>  mask;
   //std::vector<Float_t> output;

   GetInput(ev, localInput, mask, kTRUE);
   //std::cout<<"before Transform"<<std::endl;
   TransformInputData(localInput, backTransformInput);
   //std::cout<<"after Transform"<<std::endl;
   //std::cout << "backTransformInput rows : " << backTransformInput.GetNrows() << std::endl;
   //std::cout << "backTransformInput cols : " << backTransformInput.GetNcols() << std::endl;
   backTransformOutput.ResizeTo(fAutoEncoder[currentClass]->PredictDecodedOutput(backTransformInput));
   backTransformOutput = fAutoEncoder[currentClass]->PredictDecodedOutput(backTransformInput);
   //std::cout << "backTransformOutput rows : " << backTransformOutput.GetNrows() << std::endl;
   //std::cout<< "backTransformOutput cols"<<backTransformOutput.GetNcols()<<std::endl;

   BackTransformOutputData(backTransformOutput, localOutput);
   SetOutput(fBackTransformedEvent, localOutput, mask, ev, kTRUE);


   return fBackTransformedEvent;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the DAE transform for the signal and the background data
/// it uses the MakePrincipal method of ROOT's TPrincipal class

void TMVA::VariableDAETransform::TrainOnExampleData( const std::vector< Event*>& events )
{
   //size_t BatchSize = 1;
   size_t InputDepth = 1;     // Just put 1 here
   size_t InputHeight = 1;
   size_t InputWidth = 1;
   size_t BatchDepth = 1;
   size_t BatchHeight = 1;
   size_t BatchWidth = 1;
   DNN::ELossFunction fJ = DNN::ELossFunction::kCrossEntropy;
   DNN::EInitialization fI = DNN::EInitialization::kUniform;
   DNN::ERegularization fR = DNN::ERegularization::kNone;
   Scalar_t fWeightDecay = 0.0;
   bool isTraining = false;

   std::vector<size_t> numHiddenUnitsPerLayer = {2};
   Scalar_t learningRate = 0.1;
   Scalar_t corruptionLevel = 0.2;     // between 0.1 and 0.3 
   Scalar_t dropoutProbability = 0.2;
   size_t epochs = 15000;
   DNN::EActivationFunction activation;
   bool applyDropout = true;
   activation = DNN::EActivationFunction::kSigmoid;

   numCompressedUnits = numHiddenUnitsPerLayer.back();


   UInt_t nvars = 0, ntgts = 0, nspcts = 0;
   CountVariableTypes( nvars, ntgts, nspcts );
   if( nvars>0  && ntgts>0 )
      Log() << kFATAL << "Variables and targets cannot be mixed in DeepAutoEncoder transformation." << Endl;



   //const Int_t inputSize = fGet.size();

   // if we have more than one class, add another PCA analysis which combines all classes
   const UInt_t nCls = GetNClasses();
   const UInt_t numDAE = (nCls<=1) ? 1 : nCls+1;
// ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

   // !! Not normalizing and not storing input data, for performance reasons. Should perhaps restore normalization.
   // But this can be done afterwards by adding a normalisation transformation (user defined)

   size_t visibleUnits = events[0]->GetValues().size();
   size_t numEvents = events.size();

   numHiddenUnitsPerLayer.clear();
   numHiddenUnitsPerLayer.push_back(visibleUnits);
   numHiddenUnitsPerLayer.push_back(static_cast<size_t>(std::round(static_cast<double>(visibleUnits)/2.)));   // 2




   std::vector<Float_t> bareinput;
   std::vector<Char_t>  mask;

   input.clear();
   //std::cout << "Nclasses : " << nCls << std::endl;
   //std::cout << "Initializing input vector ";
   for (unsigned int i=0; i<numDAE; i++)
   {
      input.emplace_back(std::vector<Matrix_t>(0));
      //std::cout << input[i].size() << " ";
   }
   //std::cout << "Initialization finished" << std::endl;
   //std::cout << input.size() << std::endl;
   std::vector<size_t> evtsPerClass(nCls, -1);     // Initialising with -1 so that the first element inserted coincides with index 0.
   //std::cout << evtsPerClass.size() << std::endl;

   Matrix_t transformedInput(visibleUnits, 1);
   for ( unsigned int i = 0; i<numEvents; i++ )
   {

      const Event* ev = events[i];        // Why this? Can't we just pass events[i] in the function?
      UInt_t cls = ev->GetClass();

      input[cls].emplace_back(visibleUnits, 1);
      //totalInput.emplace_back(visibleUnits, 1);
      if (nCls > 1)
      {
         input[input.size()-1].emplace_back(visibleUnits, 1);
      }
      evtsPerClass[cls]++;
      //std::cout << cls << " " << evtsPerClass[cls] << std::endl;


      Bool_t hasMaskedEntries = GetInput( ev, bareinput, mask );

      if (hasMaskedEntries){
         Log() << kWARNING << "Print event which triggers an error" << Endl;
         std::ostringstream oss;
         ev->Print(oss);
         Log() << oss.str();
         Log() << kFATAL << "Masked entries found in event read in when calculating the principal components for the PCA transformation." << Endl;
      }

      TransformInputData(bareinput, transformedInput);     // Seems to be working

      input[cls][evtsPerClass[cls]] = transformedInput;
      if (nCls > 1)
      {
        input[input.size()-1][i] = transformedInput;
      }
      //std::cout << "Transformations succeded " << std::endl;
   }



   for (UInt_t i=0; i<numDAE; i++)
   {
      std::cout << "input size " << i << " : " << input[i].size() << std::endl;
      fAutoEncoder.push_back( new TMVA::DNN::TDeepAutoEncoder<Architecture_t>(input[i].size(), InputDepth, InputHeight, InputWidth, BatchDepth, BatchHeight, BatchWidth, fJ, fI, fR, fWeightDecay, isTraining) );
      std::cout << "Training autoencoder " << i << std::endl;
      fAutoEncoder.at(i)->PreTrain(input[i], numHiddenUnitsPerLayer, learningRate, corruptionLevel, dropoutProbability, epochs, activation, applyDropout);
      if (i==0) {fAutoEncoder.at(0)->WriteToXML("/home/giuseppe/rootauto/testing/autoencoder1config.xml"); } 
   }


   Char_t varType = 's';
   if (nvars > 0)    // If the transfomation was done on variables
   {
      varType = 'v';
   }
   else if (ntgts > 0)  // If the transformation was done on targets
   {
      varType = 't';
   }
   else
   {
      Log() << kFATAL << "No variables or only spectators, cannot perform autoencoder transformation. " << Endl;
   }

   fPut.clear();
   for (size_t i=0; i<(size_t)numCompressedUnits; i++)
   {
      fPut.push_back(std::pair<Char_t,UInt_t>(varType, i));
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
      catch(std::out_of_range&  excpt * ){ // happens when an event is transformed which does not yet have the targets calculated (in the application phase)
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
      //input.emplace_back(visibleUnits, 1);
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
   remoteInput.ResizeTo(numVar, 1);    //Matrix_t localInput(numVar, 1);
   for ( unsigned int i = 0; i<numVar; i++ )
   {
      remoteInput(i, 0) = localEvent[i];
   }
   //std::cout << "Matrix copy" << std::endl;     // Works now
}

void TMVA::VariableDAETransform::BackTransformOutputData( const Matrix_t& autoencoderOutput, std::vector<Float_t>& vec) const
{
   /*for (unsigned int i=0; i<numEvents; i++)
   {
      output.emplace_back(hiddenUnits, 1);
   }*/
   std::vector<Float_t> outputVector;
   vec.clear();
   if (autoencoderOutput.GetNcols()<2)
   {
      for (size_t i=0; i< (size_t)autoencoderOutput.GetNrows(); i++)
         vec.push_back(autoencoderOutput(i, 0));
   }

}


////////////////////////////////////////////////////////////////////////////////
/// write mean values to stream

void TMVA::VariableDAETransform::WriteTransformationToStream( std::ostream& o ) const
{
   for (Int_t sbType=0; sbType<2; sbType++) {
      o << "Deep autoencoder output values " << std::endl;
      //const TVectorD* means = fMeanValues[sbType];
      //o << (sbType==0 ? "Signal" : "Background") << " " << means->GetNrows() << std::endl;
      //for (Int_t row = 0; row<means->GetNrows(); row++) {
      //   o << std::setprecision(12) << std::setw(20) << (*means)[row];
      //}
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
   /*if (part==1) {
      fout << std::endl;
      fout << "   void Transform_"<<trCounter<<"( const double*, double*, int ) const;" << std::endl;
      fout << "   double input_"<<trCounter<<"["<<numC<<"]["
           << input[0].GetNrows()   << "];" << std::endl;   // mean values
      fout << "   double output_"<<trCounter<<"["<<numC<<"]["
           << input[0].GetNrows() << "]["
           << output[0].GetNcols() <<"];" << std::endl;   // eigenvectors
      fout << std::endl;
   }*/

   if (part==2) {

      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::Transform_"<<trCounter<<"( const Event* const ev, Int_t cls ) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   numDAE = " << fAutoEncoder.size() << ";" << std::endl; 
      fout << std::endl; 
      fout << "if (cls < 0 || cls >= (int) input.size()) cls = input.size()-1;" << std::endl; 
      fout << "if (fTransformedEvent==0 ) {" << std::endl; 
      fout << "fTransformedEvent = new Event();" << std::endl; 
      fout << "}" << std::endl; 
      fout << "for (unsigned int i=0; i<numDAE; i++) {" << std::endl; 
      fout << "   fAutoEncoder.push_back(TDeepAutoEncoder('/filepath'+std::to_tring(i) ));" << std::endl; 
      fout << "}" << std::endl; 
      fout << "Matrix_t transformedEvent, encodedEvent;" << std::endl; 
      fout << "std::vector<Float_t> localInput, localOutput;" << std::endl; 
      fout << "std::vector<Char_t>  mask;" << std::endl; 
      fout << "encodedEvent.ResizeTo(numCompressedUnits, 1);" << std::endl; 
      fout << "Bool_t hasMaskedEntries = GetInput( ev, localInput, mask );" << std::endl; 
      fout << "if( hasMaskedEntries ){ // targets might be masked (for events where the targets have not been computed yet)" << std::endl; 
      fout << "   UInt_t numMasked = std::count(mask.begin(), mask.end(), (Char_t)kTRUE);" << std::endl; 
      fout << "   UInt_t numOK     = std::count(mask.begin(), mask.end(), (Char_t)kFALSE);" << std::endl; 
      fout << "   if( numMasked>0 && numOK>0 ){" << std::endl; 
      fout << "      Log() << kFATAL << 'You mixed variables and targets in the Deep Autoencoder transformation. This is not possible.' << Endl;" << std::endl; 
      fout << "   }" << std::endl; 
      fout << "   SetOutput( fTransformedEvent, localInput, mask, ev );" << std::endl; 
      fout << "   return fTransformedEvent;" << std::endl; 
      fout << "}" << std::endl; 
      fout << "TransformInputData(localInput, transformedEvent);" << std::endl; 
      fout << "encodedEvent = fAutoEncoder[cls]->PredictEncodedOutput(transformedEvent);" << std::endl; 
      fout << "BackTransformOutputData(encodedEvent, localOutput);" << std::endl; 
      fout << "SetOutput( fTransformedEvent, localOutput, mask, ev );" << std::endl; 
      fout << "}" << std::endl; 
      
   }
}
void TMVA::VariableDAETransform::ReadFromFile()
{
   std::ifstream weightsfile, hiddenbiasesfile, visiblebiasesfile, info;
   std::string line;
   info.open("layersInfo.txt");
   info >> line;
   size_t layers = std::stoi(line);

   for(size_t i=0; i < layers; i++)
   {
      weightsfile.open("Weights"+std::to_string(layers)+".txt");
      hiddenbiasesfile.open("HiddenBiases"+std::to_string(layers)+".txt");
      visiblebiasesfile.open("VisibleBiases"+std::to_string(layers)+".txt");
   }

}
