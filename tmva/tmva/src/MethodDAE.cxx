// @(#)root/tmva $Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis *
 * Package: TMVA *
 * Class  : MethodDAE *
 * Web    : http://tmva.sourceforge.net *
 *                                                                                *
 * Description: *
 *      A neural network implementation *
 *                                                                                *
 * Authors (alphabetical): *
 *                                                                                *
 *                                                                                *
 * Copyright (c) 2005-2015: *
 *                                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without *
 * modification, are permitted according to the terms listed in LICENSE *
 * (http://tmva.sourceforge.net/LICENSE) *
 **********************************************************************************/

/*! \class TMVA::MethodDAE
\ingroup TMVA
 Implementation.
*/
#include "TMVA/MethodDAE.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/Types.h"

#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/Parsing.h"
#include "TMVA/DNN/DAE/DenoiseAE.h"
#include "TMVA/DNN/Net.h"
#include "TMVA/DNN/Architectures/Reference.h"
//#include "TMVA/DNN/TestLogisticRegressionLayer.h"
//#include "TMVA/DNN/TestUtilsAE.h"



#include "TMVA/NeuralNet.h"
#include <iostream>

REGISTER_METHOD(DAE)
ClassImp(TMVA::MethodDAE)

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;

namespace TMVA {
  //using namespace DNNParsing;

////////////////////////////////////////////////////////////////////////////////
/// standard constructor
TMVA::MethodDAE::MethodDAE(const TString &jobName, const TString &methodTitle,
                           DataSetInfo &theData, const TString &theOption)
    : MethodBase(jobName, Types::kDAE, methodTitle, theData, theOption),
      fWeightInitialization(), fOutputFunction(), fLayoutString(),
      fErrorStrategy(), fTrainingStrategyString(),
      fWeightInitializationString(), fArchitectureString(), fResume(false),
      fTrainingSettings()
{
  // Nothing to do here
  fAnalysisType = Types::EAnalysisType(Types::kUnsupervised);
}

////////////////////////////////////////////////////////////////////////////////
/// constructor from a weight file
TMVA::MethodDAE::MethodDAE(DataSetInfo &theData, const TString &theWeightFile)
    : MethodBase(Types::kDAE, theData, theWeightFile), fWeightInitialization(),
      fOutputFunction(), fLayoutString(), fErrorStrategy(),
      fTrainingStrategyString(), fWeightInitializationString(),
      fArchitectureString(), fResume(false), fTrainingSettings() {
  // Nothing to do here
  fAnalysisType = Types::EAnalysisType(Types::kUnsupervised);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::MethodDAE::~MethodDAE() {
  delete net;
}

////////////////////////////////////////////////////////////////////////////////
/// default initializations
void TMVA::MethodDAE::Init() {
  // Nothing to do here
  std::cout << "MethodDAE initialized successfully! " << std::endl;

   auto &dsi = this->DataInfo();
  auto numClasses = dsi.GetNClasses();
  for(UInt_t i=0; i<numClasses; ++i)
  {
    if(dsi.GetWeightExpression(i) != TString(""))
    {
      Log() << kERROR << "Currently event weights are not considered properly by this method." << Endl;
      break;
    }
  }

}

////////////////////////////////////////////////////////////////////////////////
/// Options to be set in the option string
void TMVA::MethodDAE::DeclareOptions() {                        // Change this
  DeclareOptionRef(fLayoutString="SOFTSIGN|(N+100)*2,LINEAR",
                                  "Layout",
                                  "Layout of the network.");

   DeclareOptionRef(fErrorStrategy="CROSSENTROPY",
                    "ErrorStrategy",
                    "Loss function: Mean squared error (regression)"
                    " or cross entropy (binary classification).");
   AddPreDefVal(TString("CROSSENTROPY"));
   AddPreDefVal(TString("SUMOFSQUARES"));
   AddPreDefVal(TString("MUTUALEXCLUSIVE"));

   DeclareOptionRef(fWeightInitializationString="XAVIER",
                    "WeightInitialization",
                    "Weight initialization strategy");
   AddPreDefVal(TString("XAVIER"));
   AddPreDefVal(TString("XAVIERUNIFORM"));

   DeclareOptionRef(fArchitectureString = "STANDARD", "Architecture", "Which architecture to perform the training on.");
   AddPreDefVal(TString("STANDARD"));
   AddPreDefVal(TString("CPU"));
   AddPreDefVal(TString("GPU"));
   AddPreDefVal(TString("OPENCL"));

   DeclareOptionRef(
       fTrainingStrategyString = "LearningRate=1e-1,"
                                 "Repetitions=3,"
                                 "ConvergenceSteps=50,"
                                 "BatchSize=30,"
                                 "TestRepetitions=7,"
                                 "Renormalize=L2,"
                                 "Corruption=0.0,"
                                 "DropRepetitions=5|LearningRate=1e-4,"
                                 "Repetitions=3,"
                                 "ConvergenceSteps=50,"
                                 "BatchSize=20,"
                                 "TestRepetitions=7,"
                                 "WeightDecay=0.001,"
                                 "DropRepetitions=5,"
                                 "Multithreading=True",
                                 "TrainingStrategy",
                                 "Defines the training strategies.");
}

void TMVA::MethodDAE::ProcessOptions() {
  if (IgnoreEventsWithNegWeightsInTraining()) {             // Shall we keep this?
      Log() << kINFO
            << "Will ignore negative events in training!"
            << Endl;
   }

   //constexpr Architecture_t architecture = TStandard<>;
   if (fArchitectureString == "STANDARD") {
      /*Log() << kERROR << "The STANDARD architecture has been deprecated. "
                         "Please use Architecture=CPU or Architecture=CPU."
                         "See the TMVA Users' Guide for instructions if you "
                         "encounter problems."
            << Endl;
      Log() << kFATAL << "The STANDARD architecture has been deprecated. "
                         "Please use Architecture=CPU or Architecture=CPU."
                         "See the TMVA Users' Guide for instructions if you "
                         "encounter problems."
            << Endl;*/

   }

   if (fArchitectureString == "OPENCL") {
      Log() << kERROR << "The OPENCL architecture has not been implemented yet. "
                         "Please use Architecture=CPU or Architecture=CPU for the "
                         "time being. See the TMVA Users' Guide for instructions "
                         "if you encounter problems."
            << Endl;
      Log() << kFATAL << "The OPENCL architecture has not been implemented yet. "
                         "Please use Architecture=CPU or Architecture=CPU for the "
                         "time being. See the TMVA Users' Guide for instructions "
                         "if you encounter problems."
            << Endl;
   }

   if (fArchitectureString == "GPU") {
#ifndef DNNCUDA // Included only if DNNCUDA flag is _not_ set.
      Log() << kERROR << "CUDA backend not enabled. Please make sure "
                         "you have CUDA installed and it was successfully "
                         "detected by CMAKE."
            << Endl;
      Log() << kFATAL << "CUDA backend not enabled. Please make sure "
                         "you have CUDA installed and it was successfully "
                         "detected by CMAKE."
            << Endl;
#endif // DNNCUDA
   }

   if (fArchitectureString == "CPU") {
#ifndef DNNCPU // Included only if DNNCPU flag is _not_ set.
      Log() << kERROR << "Multi-core CPU backend not enabled. Please make sure "
                         "you have a BLAS implementation and it was successfully "
                         "detected by CMake as well that the imt CMake flag is set."
            << Endl;
      Log() << kFATAL << "Multi-core CPU backend not enabled. Please make sure "
                         "you have a BLAS implementation and it was successfully "
                         "detected by CMake as well that the imt CMake flag is set."
            << Endl;
#endif // DNNCPU
   }

   //
   // Set network structure.
   //

   fLayout = TMVA::MethodDAE::ParseLayoutString (fLayoutString);
   size_t inputSize = GetNVariables ();
   size_t outputSize = 1;
   if (fAnalysisType == Types::kRegression && GetNTargets() != 0) {
      outputSize = GetNTargets();
   } else if (fAnalysisType == Types::kMulticlass && DataInfo().GetNClasses() >= 2) {
      outputSize = DataInfo().GetNClasses();
   }

   outputSize = fLayout[0];

   // Set the input size and the batch size for the net.

   //net = new TSDAE<fArchitectureString>(batchSize, inputSize, outputSize, fLayout.size(), fLayout)

   #pragma("build net here")

   //
   // Loss function and output.
   //

   /*fOutputFunction = EOutputFunction::kSigmoid;
   if (fAnalysisType == Types::kClassification)
   {
      if (fErrorStrategy == "SUMOFSQUARES") {
         fNet.SetLossFunction(ELossFunction::kMeanSquaredError);
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         fNet.SetLossFunction(ELossFunction::kCrossEntropy);
      }
      fOutputFunction = EOutputFunction::kSigmoid;
   } else if (fAnalysisType == Types::kRegression) {
      if (fErrorStrategy != "SUMOFSQUARES") {
         Log () << kWARNING << "For regression only SUMOFSQUARES is a valid "
                << " neural net error function. Setting error function to "
                << " SUMOFSQUARES now." << Endl;
      }
      fNet.SetLossFunction(ELossFunction::kMeanSquaredError);
      fOutputFunction = EOutputFunction::kIdentity;
   } else if (fAnalysisType == Types::kMulticlass) {
      if (fErrorStrategy == "SUMOFSQUARES") {
         fNet.SetLossFunction(ELossFunction::kMeanSquaredError);
      }
      if (fErrorStrategy == "CROSSENTROPY") {
         fNet.SetLossFunction(ELossFunction::kCrossEntropy);
      }
      if (fErrorStrategy == "MUTUALEXCLUSIVE") {
         fNet.SetLossFunction(ELossFunction::kSoftmaxCrossEntropy);
      }
      fOutputFunction = EOutputFunction::kSoftmax;
   }*/

   //
   // Initialization
   //

   if (fWeightInitializationString == "XAVIER") {
      fWeightInitialization = DNN::EInitialization::kGauss;
   }
   else if (fWeightInitializationString == "XAVIERUNIFORM") {
      fWeightInitialization = DNN::EInitialization::kUniform;
   }
   else {
      fWeightInitialization = DNN::EInitialization::kGauss;
   }

   //
   // Training settings.
   //

   KeyValueVector_t strategyKeyValues = ParseKeyValueString(fTrainingStrategyString,
                                                            TString ("|"),
                                                            TString (","));
   for (auto& block : strategyKeyValues) {
      TTrainingSettings settings;

      settings.convergenceSteps = DNNParsing::fetchValue(block, "ConvergenceSteps", 100);
      settings.batchSize        = DNNParsing::fetchValue(block, "BatchSize", 30);
      settings.testInterval     = DNNParsing::fetchValue(block, "TestRepetitions", 7);
      settings.weightDecay      = DNNParsing::fetchValue(block, "WeightDecay", 0.0);
      settings.learningRate         = DNNParsing::fetchValue(block, "LearningRate", 1e-5);
      //settings.momentum             = DNNParsing::fetchValue(block, "Momentum", 0.3);
      settings.corruption = DNNParsing::fetchValue(block, "Corruption", 0.0);

      //TString regularization = DNNParsing::fetchValue(block, "Regularization", TString ("NONE"));
      /*if (regularization == "L1") {
         settings.regularization = DNN::ERegularization::kL1;
      } else if (regularization == "L2") {
         settings.regularization = DNN::ERegularization::kL2;
      }*/

      TString strMultithreading = DNNParsing::fetchValue(block, "Multithreading",
                                             TString ("True"));
      if (strMultithreading.BeginsWith ("T")) {
         settings.multithreading = true;
      } else {
         settings.multithreading = false;
      }

      fTrainingSettings.push_back(settings);

      net = new DNN::DAE::TDAE<Architecture_t>(fTrainingSettings[0].batchSize, outputSize, fLayout[0]);

   }
}

////////////////////////////////////////////////////////////////////////////////
/// parse layout specification string and return a vector, each entry
/// containing the dimension and the activation function of each successive
/// layer
auto TMVA::MethodDAE::ParseLayoutString(TString layoutString)
    -> LayoutVector_t {
  // TO DO
  std::vector<Int_t> layout;

  TString delimiter = ",";

  int stringLength = layoutString.Length();
  //int delimLength=delim.Length();

  //std::cout<<"String: "<<layoutString<<"  "<<stringLength<<std::endl;
  //std::cout<<"Delim: "<<delim<<"  "<<delimLength<<std::endl;

  std::cout<<layoutString<<"  "<<stringLength<<std::endl;

  int stop=1;
  TString temp="---";
  while(stop!=-1){
      stop = layoutString.First(delimiter);

      if(stop!=-1){
          temp = layoutString(0, stop);
          std::cout<<"Substring: "<<temp<<std::endl;
          TSubString subString = layoutString( stop+delimiter.Length(), stringLength );
          layoutString=subString;
          stringLength=layoutString.Length();
      }
      else{
          stringLength=layoutString.Length();
          temp = layoutString(0, stringLength);
          std::cout<<"Substring: "<<temp<<std::endl;
      }                                                               // Works!

      layout.push_back(std::stoi(std::string(temp)));
  }



  return layout;
}

////////////////////////////////////////////////////////////////////////////////
/// What kind of analysis type can handle the DAE
Bool_t TMVA::MethodDAE::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/) {

  if (type == Types::kUnsupervised) return kTRUE;
  if (type == Types::kClassification) return kFALSE;
  if (type == Types::kMulticlass) return kFALSE;
  if (type == Types::kRegression) return kFALSE;


  return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::Train() {

  std::cout << "MethodDAE is training... " << std::endl;
  for (unsigned int i=0; i<fLayout.size(); i++) {
    std::cout << fLayout[i] << std::endl;
  }
  if (fArchitectureString == "GPU") {
    TrainGpu();
    return;
  } else if (fArchitectureString == "OpenCL") {
    Log() << kFATAL << "OpenCL backend not yet supported." << Endl;
    return;
  } else if (fArchitectureString == "CPU") {
    TrainCpu();
    return;
  }
  else {
    //TrainStd();
    /*size_t nTrainingSamples = GetEventCollection(Types::kTraining).size();
    size_t nTestSamples     = GetEventCollection(Types::kTesting).size();
    size_t trainingPhase = 1;
    for (TTrainingSettings & settings : fTrainingSettings) {
      if (fInteractive){
         fInteractive->ClearGraphs();
      }
      Log() << "Training phase " << trainingPhase << " of "
            << fTrainingSettings.size() << ":" << Endl;
      trainingPhase++;
    DataLoader_T trainingData()*/


   std::vector<Pattern> trainPattern;
   std::vector<Pattern> testPattern;

   const std::vector<TMVA::Event*>& eventCollectionTraining = GetEventCollection (Types::kTraining);
   const std::vector<TMVA::Event*>& eventCollectionTesting  = GetEventCollection (Types::kTesting);

  // For denoise layer (unsupervised learning, to come) needs unsupervised learning

   for (unsigned int t =0; t<fTrainingSettings.size(); t++) {


  TMVA::Event *event = eventCollectionTraining[0];
  const std::vector<Float_t> values = event->GetValues();
  std::cout << values.size() << std::endl;
  Int_t totalColumnsInput = values.size();
  Int_t batchSize = eventCollectionTraining.size(); //fTrainingSettings[t].batchSize;     // Fix this
  size_t totalRowsInput = eventCollectionTraining.size();
  std::cout << totalRowsInput << std::endl;
  size_t numOfFullBatches = static_cast<int>(totalRowsInput)/static_cast<int>(batchSize);
  size_t numOfRowsLastBatch = (static_cast<int>(totalRowsInput) % static_cast<int>(batchSize));
  size_t compressedUnits = 5;
  double corruptionLevel = 0.2;
  double learningRate = 0.1;
  size_t fBatchSize = totalRowsInput;

  std::cout << "Num of batches : " << numOfFullBatches + 1 << " Num of rows last batch : " << numOfRowsLastBatch << std::endl;



  std::vector<Matrix_t> input;
  for (size_t i = 0; i < numOfFullBatches; i++) {
    input.emplace_back(totalColumnsInput, 1);
  }
  //input.emplace_back(totalColumnsInput, 1);


  for (size_t i = 0; i < numOfFullBatches ; i++) {  // + 1
    for (size_t j = 0; j < (size_t)input[i].GetNrows(); j++) {
      const std::vector<Float_t> values = event->GetValues();
      for (size_t k = 0; k < (size_t)input[i].GetNcols(); k++) {
        int n(i*batchSize + j);
        input[i](j, k) = values[j];
      }
    }



      net->TrainLayer(input[i], fTrainingSettings[t].learningRate, fTrainingSettings[t].corruption);

    }
  }

  // For classification

  /*size_t totalRowsTrainOutput;
  size_t totalColumnsTrainOutput;
  std::vector<Matrix_t> TrainX;
  std::vector<Matrix_t> TrainY;
  for (unsigned int i=0; i<1*/ /*eventCollectionTraining.size()*/ /*; i++)
  {
    TMVA::Event *event = eventCollectionTraining[i];
    const std::vector<Float_t> values = event->GetValues();

    // Testing the values ...                                                   // test
    for (int j=0; j<values.size(); j++)                                         // test
    {                                                                           // test
      std::cout << values[j] << " ";                                            // test
    }                                                                           // test
    Int_t numcol = values.size();
    size_t totalRowsTrainInput = eventCollectionTraining.size();
    size_t totalColumnsTrainInput = values.size();
    totalRowsTrainOutput= totalRowsTrainInput;
    totalColumnsTrainOutput = totalColumnsTrainInput;
    TrainX.emplace_back(numcol, 1);
    TrainY.emplace_back(numcol, 1);
  }
  size_t totalRowsTestInput = eventCollectionTesting.size();
  size_t totalColumnsTestInput = 10;
  size_t totalRowsTestOutput= 2;
  size_t totalColumnsTestOutput = 2;
  std::cout << "size x : " << TrainX.size() << " size Y : " << TrainY.size() << std::endl;      // test
  Matrix_t p(totalColumnsTestOutput,1);
  Matrix_t difference(totalColumnsTestOutput,1);
  Matrix_t Weights(totalColumnsTestOutput,totalColumnsTrainInput);
  randomMatrix(Weights);
  std::cout<<"Weights "<<Weights.GetNrows()<<" "<<Weights.GetNcols()<<std::endl;
  Matrix_t Biases(totalColumnsTestOutput,1);
  for(size_t epoch=0;epoch<1000;epoch++){
  for(size_t i=0; i<totalRowsTrainInput; i++)
  {
    TMVA::Event *event = eventCollectionTraining[i];
    const std::vector<Float_t> values = event->GetValues();
    for( size_t j=0; j<(size_t)TrainX[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)TrainX[i].GetNcols(); k++)
      {
        TrainX[i](j,k) = values[j];
      }
    }
    for(size_t j=0; j<(size_t)TrainY[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)TrainY[i].GetNcols(); k++)
      {
        TrainY[i](j,k) = values[j];   // To be changed
      }
    }
    std::cout<<"crating p matrix"  << std::endl;
    testForwardLogReg<TReference<double>>(TrainX[i],p, Weights);
    std::cout<<"Adding Bias matrix to given Matrix"<<std::endl;
    testAddBiases<TReference<double>>(p,Biases);
    std::cout<<std::endl;
    std::cout<<"Given Matrix after Softmax Operation"<<std::endl;
    testSoftmaxAE<TReference<double>>(p);
    std::cout<<std::endl;
    std::cout<<"updating parameters"  << std::endl;
    testUpdateParamsLogReg<TReference<double>>(TrainX[i],TrainY[i], difference,p,
                                               Weights,Biases,fTrainingSettings.learningRate,fTrainingSettings.batchSize);
    }
  }*/




  }



}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::TrainGpu() {
  // TO DO
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::TrainCpu() {
  // TO DO
}

////////////////////////////////////////////////////////////////////////////////
Double_t TMVA::MethodDAE::GetMvaValue(Double_t * /*errLower*/,
                                      Double_t * /*errUpper*/) {
  // TO DO
  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::AddWeightsXMLTo(void *parent) const {
  /*void* nn = gTools().xmlengine().NewChild(parent, 0, "Weights");
  Int_t inputWidth = net->GetInputWidth();
  Int_t depth      = net->GetDepth();
  char  lossFunction = static_cast<char>(fNet.GetLossFunction());
  gTools().xmlengine().NewAttr(nn, 0, "InputWidth", gTools().StringFromInt(inputWidth));
  gTools().xmlengine().NewAttr(nn, 0, "Depth", gTools().StringFromInt(depth));
  gTools().xmlengine().NewAttr(nn, 0, "LossFunction", TString(lossFunction));
  gTools().xmlengine().NewAttr(nn, 0, "OutputFunction", TString(static_cast<char>(fOutputFunction)));
  for (Int_t i = 0; i < depth; i++) {
    const auto& layer = net->GetLayer(i);
    auto layerxml = gTools().xmlengine().NewChild(nn, 0, "Layer");
    int activationFunction = static_cast<int>(layer.GetActivationFunction());
    gTools().xmlengine().NewAttr(layerxml, 0, "ActivationFunction", TString::Itoa(activationFunction, 10));
    WriteMatrixXML(layerxml, "Weights", layer.GetWeights());
    WriteMatrixXML(layerxml, "Biases",  layer.GetBiases());
  }*/
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::ReadWeightsFromXML(void *rootXML) {
  /*auto netXML = gTools().GetChild(rootXML, "Weights");
  if (!netXML){
     netXML = rootXML;
  }
  //fNet.Clear();
  //fNet.SetBatchSize(1);
  size_t inputWidth, depth;
  gTools().ReadAttr(netXML, "InputWidth", inputWidth);
  gTools().ReadAttr(netXML, "Depth", depth);
  char lossFunctionChar;
  gTools().ReadAttr(netXML, "LossFunction", lossFunctionChar);
  char outputFunctionChar;
  gTools().ReadAttr(netXML, "OutputFunction", outputFunctionChar);
  fNet.SetInputWidth(inputWidth);
  fNet.SetLossFunction(static_cast<ELossFunction>(lossFunctionChar));
  fOutputFunction = static_cast<EOutputFunction>(outputFunctionChar);
  size_t previousWidth = inputWidth;
  auto layerXML = gTools().xmlengine().GetChild(netXML, "Layer");
  for (size_t i = 0; i < depth; i++) {
     TString fString;
     EActivationFunction f;
     // Read activation function.
     gTools().ReadAttr(layerXML, "ActivationFunction", fString);
     f = static_cast<EActivationFunction>(fString.Atoi());
     // Read number of neurons.
     size_t width;
     auto matrixXML = gTools().GetChild(layerXML, "Weights");
     gTools().ReadAttr(matrixXML, "rows", width);
     fNet.AddLayer(width, f);
     TMatrixT<Double_t> weights(width, previousWidth);
     TMatrixT<Double_t> biases(width, 1);
     ReadMatrixXML(layerXML, "Weights", weights);
     ReadMatrixXML(layerXML, "Biases",  biases);
     fNet.GetLayer(i).GetWeights() = weights;
     fNet.GetLayer(i).GetBiases()  = biases;
     layerXML = gTools().GetNextChild(layerXML);
     previousWidth = width;
  }*/
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::ReadWeightsFromStream(std::istream & /*istr*/) {}

////////////////////////////////////////////////////////////////////////////////
const TMVA::Ranking *TMVA::MethodDAE::CreateRanking() {
  /*fRanking = new Ranking( GetName(), "Importance" );
  for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
    fRanking->AddRank( Rank( GetInputLabel(ivar), 1.0));
  }
  return fRanking;*/
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::MethodDAE::GetHelpMessage() const {
  // TO DO
}

auto TMVA::MethodDAE::ParseKeyValueString(TString parseString,
                                          TString blockDelim,
                                          TString tokenDelim)
    -> KeyValueVector_t
{
   KeyValueVector_t blockKeyValues;
   const TString keyValueDelim ("=");

   TObjArray* blockStrings = parseString.Tokenize (blockDelim);
   TIter nextBlock (blockStrings);
   TObjString* blockString = (TObjString *) nextBlock();

   for (; blockString != nullptr; blockString = (TObjString *) nextBlock())
   {
      blockKeyValues.push_back (std::map<TString,TString>());
      std::map<TString,TString>& currentBlock = blockKeyValues.back ();

      TObjArray* subStrings = blockString->GetString ().Tokenize (tokenDelim);
      TIter nextToken (subStrings);
      TObjString* token = (TObjString*)nextToken ();

      for (; token != nullptr; token = (TObjString *)nextToken())
      {
         TString strKeyValue (token->GetString ());
         int delimPos = strKeyValue.First (keyValueDelim.Data ());
         if (delimPos <= 0)
             continue;

         TString strKey = TString (strKeyValue (0, delimPos));
         strKey.ToUpper();
         TString strValue = TString (strKeyValue (delimPos+1, strKeyValue.Length ()));

         strKey.Strip (TString::kBoth, ' ');
         strValue.Strip (TString::kBoth, ' ');

         currentBlock.insert (std::make_pair (strKey, strValue));
      }
   }
   return blockKeyValues;
}


} // namespace TMVA
