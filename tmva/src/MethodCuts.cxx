// @(#)root/tmva $Id: MethodCuts.cxx,v 1.3 2006/05/23 19:35:06 brun Exp $ 
// Author: Andreas Hoecker, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodCuts                                                      *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//Begin_Html
/*
  Multivariate optimisation of signal efficiency for given background  
  efficiency, applying rectangular minimum and maximum requirements.

  <p>
  <font size="-1">
  Other optimisation criteria, such as maximising the signal significance-
  squared, S^2/(S+B), with S and B being the signal and background yields, 
  correspond to a particular point in the optimised background rejection 
  versus signal efficiency curve. This working point requires the knowledge 
  of the expected yields, which is not the case in general. Note also that 
  for rare signals, Poissonian statistics should be used, which modifies 
  the significance criterion. 
  </font>

  <p>
  The rectangular cut of a volume in the variable space is performed using 
  a binary tree to sort the training events. This provides a significant 
  reduction in computing time (up to several orders of magnitudes, depending
  on the complexity of the problem at hand).

  <p>
  Technically, optimisation is achieved in TMVA by two methods:

  <ol>
  <li>Monte Carlo generation using uniform priors for the lower cut value, 
  and the cut width, thrown within the variable ranges. 

  <li>A Genetic Algorithm (GA) searches for the optimal ("fittest") cut sample.
  The GA is configurable by many external settings through the option 
  string. For difficult cases (such as many variables), some tuning 
  may be necessary to achieve satisfying results
  </ol>

  <p>
  <font size="-1">
  Attempts to use Minuit fits (Simplex ot Migrad) instead have not shown 
  superior results, and often failed due to convergence at local minima. 
  </font>

  <p>
  The tests we have performed so far showed that in generic applications, 
  the GA is superior to MC sampling, and hence GA is the default method.
  It is worthwhile to try both anyway.
*/
//End_Html


#include <stdio.h>
#include "time.h"
#include "Riostream.h"
#include "TH1F.h"
#include "TObjString.h"

#ifndef ROOT_TMVA_MethodCuts
#include "TMVA/MethodCuts.h"
#endif
#ifndef ROOT_TMVA_GeneticCuts
#include "TMVA/GeneticCuts.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif


ClassImp(TMVA::MethodCuts)

// init global variables
   TMVA::MethodCuts* TMVA::MethodCuts::fgThisCuts = NULL;

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( TString jobName, vector<TString>* theVariables,  
                              TTree* theTree, TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{ 
   // standard constructor
   // ---------------------------------------------------------------------------------- 
   // format of option string: "OptMethod:EffMethod:Option_var1:...:Option_varn"
   // "OptMethod" can be:
   //     - "GA"    : Genetic Algorithm (recommended)
   //     - "MC"    : Monte-Carlo optimization 
   // "EffMethod" can be:
   //     - "EffSel": compute efficiency by event counting
   //     - "EffPDF": compute efficiency from PDFs
   // === For "GA" method ======
   // "Option_var1++" are (see GA for explanation of parameters):
   //     - fGa_nsteps        
   //     - fGa_preCalc        
   //     - fGa_SC_steps        
   //     - fGa_SC_offsteps 
   //     - fGa_SC_factor   
   // === For "MC" method ======
   // "Option_var1" is number of random samples
   // "Option_var2++" can be 
   //     - "FMax"  : ForceMax   (the max cut is fixed to maximum of variable i)
   //     - "FMin"  : ForceMin   (the min cut is fixed to minimum of variable i)
   //     - "FSmart": ForceSmart (the min or max cut is fixed to min/max, based on mean value)
   //     - Adding "All" to "option_vari", eg, "AllFSmart" will use this option for all variables
   //     - if "option_vari" is empty (== ""), no assumptions on cut min/max are made
   // ---------------------------------------------------------------------------------- 

   InitCuts();

   TList* list  = TMVA::Tools::ParseFormatLine( fOptions );
 
   // interpret option string
   if (list->GetSize()<1) {
      fOptions = "MC:10000:";
      cout << "--- " << GetName() << ": problems with options string, using default: " 
           << fOptions << endl;
      list  = TMVA::Tools::ParseFormatLine( fOptions );
   }  

   // interpret string
   // which optimisation Method
   TString s = ((TObjString*)list->At(0))->GetString();
   s.ToUpper();
   if      (s.Contains( "MC"     )) fFitMethod = kUseMonteCarlo;
   else if (s.Contains( "GA"     )) fFitMethod = kUseGeneticAlgorithm;
   else {
      cout << "--- " << GetName() << ": unknown entry in field 0 of option string: " 
           << s << " ==> abort" << endl;
      exit(1);
   }

   if (list->GetSize() > 1) { // options are specified
      s = ((TObjString*)list->At(1))->GetString();
      s.ToUpper();

      if      (s.Contains( "EFFSEL" )) fEffMethod = kUseEventSelection; // highly recommended
      else if (s.Contains( "EFFPDF" )) fEffMethod = kUsePDFs;
      else                             fEffMethod = kUseEventSelection;
   }

   // options output
   cout << "--- " << GetName() << ": interpret options string: '" << fOptions << "'" << endl;
   printf( "--- %s: --> use optimization method: '%s'\n", 
           GetName(), (fFitMethod == kUseMonteCarlo) ? "Monte Carlo" : "Genetic Algorithm" );
   printf( "--- %s: --> use efficiency computation method: '%s'\n", 
           GetName(), (fEffMethod == kUseEventSelection) ? "Event Selection" : "PDF" );

   // -----------------------------------------------------------------------------------
   // interpret for MC use  
   //
   switch (fFitMethod) {

   case kUseMonteCarlo:

      if (list->GetSize() > 2) {

         s = ((TObjString*)list->At(2))->GetString();
         fNRandCuts = atoi( s );
         if (fNRandCuts <= 1) {
            cout << "--- " << GetName() << ": invalid number of MC events: " <<  fNRandCuts 
                 << " in field 2 of option string: " << s << " ==> abort" << endl;
            exit(1);
         }
      }
    
      cout << "--- " << GetName() << ": generate " << fNRandCuts << " random cut samples"
           << endl;
  
      if (list->GetSize() > 3) { // options are specified

         s = ((TObjString*)list->At(3))->GetString();
         s.ToUpper();
         if (s.Contains( "ALL" )) { // one option sets all the others
            FitParameters theFitP = kNotEnforced;
            if      (s.Contains( "FMAX"   )) theFitP = kForceMax;
            else if (s.Contains( "FMIN"   )) theFitP = kForceMin;
            else if (s.Contains( "FSMART" )) theFitP = kForceSmart;
            else if (s.Contains( "FVERYSMART" )) theFitP = kForceVerySmart;
            else {
               cout << "--- " << GetName() << ": unknown fit parameter option "
                    << " in field 2 of option string: " << s << " ==> abort" << endl;
               exit(1);
            }
            for (Int_t ivar=0; ivar<fNvar; ivar++) (*fFitParams)[ivar] = theFitP;

            if (theFitP != kNotEnforced) 
               cout << "--- " << GetName() << ": use 'smart' cuts" << endl;
         }
         else { // individual options
            for (Int_t ivar=0; ivar<fNvar; ivar++) {
               if (list->GetSize() >= 3+ivar) {
                  s = ((TObjString*)list->At(2+ivar))->GetString();
                  s.ToUpper();
                  FitParameters theFitP = kNotEnforced;
                  if      (s == "" || s == "NOTENFORCED") theFitP = kNotEnforced;
                  else if (s.Contains( "FMAX"   )) theFitP = kForceMax;
                  else if (s.Contains( "FMIN"   )) theFitP = kForceMin;
                  else if (s.Contains( "FSMART" )) theFitP = kForceSmart;
                  else if (s.Contains( "FVERYSMART" )) theFitP = kForceVerySmart;
                  else {
                     cout << "--- " << GetName() << ": unknown fit parameter option "
                          << " in field " << ivar+3 << " (var: " << ivar 
                          << " of option string: " << s << " ==> abort" << endl;
                     exit(1);
                  }
                  (*fFitParams)[ivar] = theFitP;

                  if (theFitP != kNotEnforced) 
                     cout << "--- " << GetName() << ": use 'smart' cuts for variable: " 
                          << "'" << (*fInputVars)[ivar] << "'" << endl;
               }        
            }
         }      
      }
      break;

      // -----------------------------------------------------------------------------------
      // interpret for GA use  
      //
   case kUseGeneticAlgorithm:
    
      if (list->GetSize() > 2) {      
         s = ((TObjString*)list->At(2))->GetString(); fGa_nsteps = atoi( s );
         if (list->GetSize() > 3) {      
            s = ((TObjString*)list->At(3))->GetString(); fGa_preCalc = atoi( s );
            if (list->GetSize() > 4) {      
               s = ((TObjString*)list->At(4))->GetString(); fGa_SC_steps = atoi( s );
               if (list->GetSize() > 5) {      
                  s = ((TObjString*)list->At(5))->GetString(); fGa_SC_offsteps = atoi( s );
                  if (list->GetSize() > 6) {      
                     s = ((TObjString*)list->At(6))->GetString(); fGa_SC_factor = atof( s );
                  }
               }
            }
         }
      }
      break;
    
   default:

      cout << "--- " << GetName() << ": Error: unknown method: " << fFitMethod 
           << " ==> abort" << endl;
      exit(1);
   }

  
   if (fFitMethod == kUseMonteCarlo) 
      printf( "--- %s: --> number of MC events to be generated: %i\n", GetName(), fNRandCuts );
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      TString theFitOption = ( ((*fFitParams)[ivar] == kNotEnforced) ? "NotEnforced" :
                               ((*fFitParams)[ivar] == kForceMin   ) ? "ForceMin"    :
                               ((*fFitParams)[ivar] == kForceMax   ) ? "ForceMax"    :
                               ((*fFitParams)[ivar] == kForceSmart ) ? "ForceSmart"  :
                               ((*fFitParams)[ivar] == kForceVerySmart ) ? "ForceVerySmart"  : "other" );
    
      printf( "--- %s: --> option for variable: %s: '%s' (#: %i)\n",
              GetName(), (const char*)(*fInputVars)[ivar], (const char*)theFitOption, 
              (Int_t)(*fFitParams)[ivar] );
   }
   // ---------------------------------------------------------------------------------- 

}

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( vector<TString> *theVariables, 
                              TString theWeightFile,  
                              TDirectory* theTargetDir )
   : TMVA::MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
   // construction from weight file
   InitCuts();
}

//_______________________________________________________________________
void TMVA::MethodCuts::InitCuts( void ) 
{
   // default initialisation called by all constructors
   fMethodName        = "Cuts";
   fMethod            = TMVA::Types::Cuts;  
   fTestvar           = fTestvarPrefix+GetMethodName();
   fConstrainType     = kConstrainEffS;
   fVarHistS          = fVarHistB = 0;                 
   fVarHistS_smooth   = fVarHistB_smooth = 0;
   fVarPdfS           = fVarPdfB = 0; 
   fFitParams         = 0;
   fEffBvsSLocal      = 0;
   fBinaryTreeS       = fBinaryTreeB = 0;
   fEffSMin           = 0;
   fEffSMax           = 0; 

   // MC defaults
   fNRandCuts         = 100000;

   // GA defaults
   fGa_preCalc        = 3;
   fGa_SC_steps       = 10;
   fGa_SC_offsteps    = 5;
   fGa_SC_factor      = 0.95;
   fGa_nsteps         = 30;

   fgThisCuts         = this;

   // vector with fit results
   fNpar      = 2*fNvar;
   fRangeSign = new vector<Int_t>   ( fNvar );
   fMeanS     = new vector<Double_t>( fNvar ); 
   fMeanB     = new vector<Double_t>( fNvar ); 
   fRmsS      = new vector<Double_t>( fNvar );  
   fRmsB      = new vector<Double_t>( fNvar );  
   fXmin      = new vector<Double_t>( fNvar );  
   fXmax      = new vector<Double_t>( fNvar );  

   // get the variable specific options, first initialize default
   fFitParams = new vector<FitParameters>( fNvar );
   for (Int_t ivar=0; ivar<fNvar; ivar++) (*fFitParams)[ivar] = kNotEnforced;

   fTrandom   = new TRandom( 0 ); // set seed
   fFitMethod = kUseMonteCarlo;
   fTestSignalEff = -1;

   // create LUT for cuts
   fCutMin = new Double_t*[fNvar];
   fCutMax = new Double_t*[fNvar];
   for (Int_t i=0;i<fNvar;i++) {
      fCutMin[i] = new Double_t[fNbins];
      fCutMax[i] = new Double_t[fNbins];
   }
  
   // init
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      for (Int_t ibin=0; ibin<fNbins; ibin++) {
         fCutMin[ivar][ibin] = 0;
         fCutMax[ivar][ibin] = 0;
      }
   }

   fTmpCutMin = new Double_t[fNvar];
   fTmpCutMax = new Double_t[fNvar];
}

//_______________________________________________________________________
TMVA::MethodCuts::~MethodCuts( void )
{
   // destructor
   if (Verbose()){
      cout << "--- TMVA::MethodCuts: Destructor called " << endl;
   }
   delete fRangeSign;
   delete fTrandom;
   delete fMeanS;
   delete fMeanB;
   delete fRmsS;
   delete fRmsB;
   delete fXmin;
   delete fXmax;  
   for (Int_t i=0;i<fNvar;i++) {
      if (fCutMin[i] != NULL) delete [] fCutMin[i];
      if (fCutMax[i] != NULL) delete [] fCutMax[i];
   }

   delete[] fCutMin;
   delete[] fCutMax;

   delete[] fTmpCutMin;
   delete[] fTmpCutMax;

   if (NULL != fBinaryTreeS) delete fBinaryTreeS;
   if (NULL != fBinaryTreeB) delete fBinaryTreeB;
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::GetMvaValue( TMVA::Event *e )
{
   // cut evaluation: returns 1.0 if event passed, 0.0 otherwise
   
   // sanity check
   if (fCutMin == NULL || fCutMax == NULL || fNbins == 0) {
      cerr << "--- " << GetName() << "::Eval_Cuts: Fatal Error: fCutMin/Max have zero pointer. "
           << "Did you book Cuts ? ==> abort" << endl;
      exit(1);
   }

   // sanity check
   if (fTestSignalEff > 0) {  
      // get efficiency bin
      Int_t ibin = int((fTestSignalEff - fEffSMin)/(fEffSMax - fEffSMin)*Double_t(fNbins));
      if (ibin < 0       ) ibin = 0;
      if (ibin >= fNbins) ibin = fNbins - 1;
    
      Bool_t passed = kTRUE;
      for (size_t ivar=0; ivar<e->GetData().size(); ivar++) {
         passed &= (e->GetData()[ivar] >= fCutMin[ivar][ibin]) && (e->GetData()[ivar] <= fCutMax[ivar][ibin]);
      }
      return passed ? 1. : 0. ;
   }
   else return 0;
}

//_______________________________________________________________________
void  TMVA::MethodCuts::Train( void )
{
   // training method: here the cuts are optimised for the training sample
   
   // perform basic sanity chacks
   if (!SanityChecks()) {
      cout << "--- " << GetName() << ": Error: Basic sanity checks failed ==> abort"
           << endl;
      exit(1);
   }

   if (fEffMethod == kUsePDFs) CreateVariablePDFs(); // create PDFs for variables

   // get background efficiency for which the signal efficiency
   // ought to be maximized
   fConstrainType = kConstrainEffS;

   // create binary trees (global member variables) for signal and background
   fBinaryTreeS = new TMVA::BinarySearchTree();
   fBinaryTreeS->Fill( fTrainingTree, fInputVars, 1 );
   fBinaryTreeB = new TMVA::BinarySearchTree();
   fBinaryTreeB->Fill( fTrainingTree, fInputVars, 0 );

   // init basic statistics
   TObjArrayIter branchIter( fTrainingTree->GetListOfBranches(), kIterForward );
   TBranch*      branch = 0;
   Int_t         ivar   = -1;
   const int nBranches  = ( fTrainingTree->GetListOfBranches() != 0 ? 
                            fTrainingTree->GetListOfBranches()->GetSize() : 0 );
   TString* branchName  = new TString[nBranches];
   Float_t* branchVar   = new Float_t[nBranches];
   Int_t    theType;

   vector<TH1F*> signalDist, bkgDist;

   while ((branch = (TBranch*)branchIter.Next()) != 0) {
      // note: allowed are only variables with minimum and maximum cut
      //       i.e., no distinct cut regions are supported
      if ((TString)branch->GetName() == "type") {
         fTrainingTree->SetBranchAddress( branch->GetName(), &theType );
      }
      else {
         ++ivar;
         branchName[ivar] = branch->GetName();
         fTrainingTree->SetBranchAddress( branchName[ivar],  &branchVar[ivar] );

         // determine mean and rms to obtain appropriate starting values
         TMVA::Tools::ComputeStat( fTrainingTree, branchName[ivar],
                                   (*fMeanS)[ivar], (*fMeanB)[ivar], 
                                   (*fRmsS)[ivar], (*fRmsB)[ivar], 
                                   (*fXmin)[ivar], (*fXmax)[ivar] );
      
         // I want to use these distributions later to steer the MC-Method a bit into the 
         // direction where the difference in the distributions for BKG and Signal are largest
         TString name = Form( "sigDistVar%d",ivar );
         signalDist.push_back( (TH1F*)TMVA::Tools::projNormTH1F( fTrainingTree, branchName[ivar], name, 50,
                                                                 (*fXmin)[ivar], (*fXmax)[ivar],
                                                                 "type==1" ) );
         name = Form( "bkgDistVar%d",ivar );
         bkgDist.push_back( (TH1F*)TMVA::Tools::projNormTH1F( fTrainingTree, branchName[ivar], name,50,
                                                              (*fXmin)[ivar],(*fXmax)[ivar],
                                                              "type==0" ) );
  
         if ((*fInputVars)[ivar] != branchName[ivar]) {
            cout << "Error in: " << GetName() << "::Train: mismatch in variables ==> abort: "
                 << ivar << " " << (*fInputVars)[ivar] << " " << branchName[ivar]
                 << endl;
            exit(1);
         }
      }
   }

   delete[] branchName;
   delete[] branchVar;

   // determine eff(B) versus eff(S) plot
   fConstrainType = kConstrainEffS;

   Int_t ibin=0;
   fEffBvsSLocal = new TH1F( fTestvar + "_effBvsSLocal", 
                             TString(GetName()) + " efficiency of B vs S", 
                             fNbins, 0.0, 1.0 );

   // init
   for (ibin=1; ibin<=fNbins; ibin++) fEffBvsSLocal->SetBinContent( ibin, -0.1 );

   // --------------------------------------------------------------------------
   if (fFitMethod == kUseMonteCarlo) {
    
      // generate MC cuts
      Double_t * cutMin = new Double_t[fNvar];
      Double_t * cutMax = new Double_t[fNvar];
    
      // MC loop
      cout << "--- " << GetName() << ": Generating " << fNRandCuts 
           << " cycles (random cuts) ... patience please" << endl;

      Int_t nBinsFilled=0, nBinsFilledAt=0;

      // timing of MC
      TMVA::Timer timer( fNRandCuts, GetName() ); 

      for (Int_t imc=0; imc<fNRandCuts; imc++) {

         // generate random cuts
         for (Int_t ivar=0; ivar<fNvar; ivar++) {

            FitParameters fitParam = (*fFitParams)[ivar];

            if (fitParam == kForceSmart) {
               if ((*fMeanS)[ivar] > (*fMeanB)[ivar]) fitParam = kForceMax;
               else                                   fitParam = kForceMin;          
            }

            if (fitParam == kForceMin) 
               cutMin[ivar] = (*fXmin)[ivar];
            else
               cutMin[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - (*fXmin)[ivar]) + (*fXmin)[ivar];

            if (fitParam == kForceMax) 
               cutMax[ivar] = (*fXmax)[ivar];
            else
               cutMax[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - cutMin[ivar]   ) + cutMin[ivar];
        
            if (fitParam == kForceVerySmart){
               // generate random cut parameters gaussian distrubuted around the variable values
               // where the difference between signal and background is maximal
          
               // get the variable distributions:
               cutMin[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - (*fXmin)[ivar]) + (*fXmin)[ivar];
               cutMax[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - cutMin[ivar]   ) + cutMin[ivar];
               // ..... to be continued (Helge)
            }

            if (cutMax[ivar] < cutMin[ivar]) {
               cout << "--- " << GetName() << ": Error in ::Train: mismatch with cuts ==> abort"
                    << endl;
               exit(1);
            }
         }

         // event loop
         Double_t effS = 0, effB = 0;
         GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);
    
         // determine bin
         Int_t    ibinS = (Int_t)(effS*Float_t(fNbins) + 1);
         if (ibinS < 1     ) ibinS = 1;
         if (ibinS > fNbins) ibinS = fNbins;
      
         // linear extrapolation 
         // (not done at present --> MC will be slightly biased !
         //  the bias increases with the bin width)
         Double_t effBH = fEffBvsSLocal->GetBinContent( ibinS );

         // preliminary best event -> backup
         if (effBH < 0 || effBH > effB) {
            fEffBvsSLocal->SetBinContent( ibinS, effB );
            for (Int_t ivar=0; ivar<fNvar; ivar++) {
               fCutMin[ivar][ibinS-1] = cutMin[ivar]; // bin 1 stored in index 0
               fCutMax[ivar][ibinS-1] = cutMax[ivar];
            }
         }

         // some output to make waiting less boring
         Int_t nout = 1000;
         if ((Int_t)imc%nout == 0  || imc == fNRandCuts-1) {
            Int_t nbinsF = 0, ibin_;
            for (ibin_=0; ibin_<fNbins; ibin_++)
               if (fEffBvsSLocal->GetBinContent( ibin_ +1   ) >= 0) nbinsF++;
            if (nBinsFilled!=nbinsF) {
               nBinsFilled = nbinsF;
               nBinsFilledAt = imc;
            }
        
            timer.DrawProgressBar( imc );
            if (imc == fNRandCuts-1 ) 
               printf( "--- %s: fraction of efficiency bins filled: %3.1f\n",
                       GetName(), nbinsF/Float_t(fNbins) );
         }
      } // end of MC loop

      if (this->Verbose()){
         cout << "--- TMVA::MethodCuts: fraction of filled eff. bins did not increase" 
              << " anymore after "<< nBinsFilledAt << " cycles" << endl;
      }

      // get elapsed time
      cout << "--- " << GetName() << ": elapsed time: " << timer.GetElapsedTime() << endl;    

      delete[] cutMin;
      delete[] cutMax;

   }
   // --------------------------------------------------------------------------
   else if (fFitMethod == kUseGeneticAlgorithm) {

      // ranges
      vector<LowHigh_t*> ranges;
    
      for (Int_t ivar=0; ivar<fNvar; ivar++) {
         (*fRangeSign)[ivar] = +1;    
         ranges.push_back( new LowHigh_t( (*fXmin)[ivar], (*fXmax)[ivar] ) );
         ranges.push_back( new LowHigh_t( 0, (*fXmax)[ivar] - (*fXmin)[ivar] ) );
      }

      TMVA::GeneticCuts *bestResultsStore = new TMVA::GeneticCuts( 0, ranges ); 
      TMVA::GeneticCuts *bestResults      = new TMVA::GeneticCuts( 0, ranges );

      cout << "--- " << GetName() << ": GA: entree, please be patient ..." << endl;

      // timing of MC
      TMVA::Timer timer1( fGa_preCalc*fNbins, GetName() ); 

      // precalculation
      for (Int_t preCalc = 0; preCalc < fGa_preCalc; preCalc++) {

         for (Int_t ibin=1; ibin<=fNbins; ibin++) {

            timer1.DrawProgressBar( ibin + preCalc*fNbins );

            fEffRef = fEffBvsSLocal->GetBinCenter( ibin );

            // ---- perform series of fits to achieve best convergence

            // "m_ga_spread" times the number of variables
            TMVA::GeneticCuts ga( ranges.size() * 10, ranges ); 

            ga.GetGeneticPopulation().AddPopulation( &bestResults->GetGeneticPopulation() );
            ga.CalculateFitness();
            ga.GetGeneticPopulation().TrimPopulation();

            do {
               ga.Init();
               ga.CalculateFitness();
               ga.SpreadControl( fGa_SC_steps, fGa_SC_offsteps, fGa_SC_factor );
            } while (!ga.HasConverged( Int_t(fGa_nsteps*0.67), 0.0001 ));
        
            bestResultsStore->GetGeneticPopulation().GiveHint( ga.GetGeneticPopulation().GetGenes( 0 )->GetFactors() );
         }
         delete bestResults;
         bestResults = bestResultsStore;
         bestResultsStore = new TMVA::GeneticCuts( 0, ranges );
                
      }

      bestResults->Init();

      // main run
      cout << "--- " << GetName() << ": GA: start main course                                    " 
           << endl;

      // timing of MC
      TMVA::Timer timer2( fNbins, GetName() ); 

      Double_t * cutMin = new Double_t[fNvar];
      Double_t * cutMax = new Double_t[fNvar];
      vector<Double_t> par(2*fNvar);
      for (ibin=1; ibin<=fNbins; ibin++) {

         timer2.DrawProgressBar( ibin );
      
         fEffRef = fEffBvsSLocal->GetBinCenter( ibin );

         // ---- perform series of fits to achieve best convergence

         TMVA::GeneticCuts ga( ranges.size() * 10, ranges ); // 10 times the number of variables
         ga.SetSpread( 0.1 );
         ga.GetGeneticPopulation().AddPopulation( &bestResults->GetGeneticPopulation() );
         ga.CalculateFitness();
         ga.GetGeneticPopulation().TrimPopulation();
         do {
            ga.Init();
            ga.CalculateFitness();
            ga.SpreadControl( fGa_SC_steps, fGa_SC_offsteps, fGa_SC_factor );
         } while (!ga.HasConverged( fGa_nsteps, 0.00001 ));

         Int_t n;

         n = 0;
         vector< Double_t >::iterator vec = ga.GetGeneticPopulation().GetGenes( 0 )->GetFactors().begin();
         for (; vec < ga.GetGeneticPopulation().GetGenes( 0 )->GetFactors().end(); vec++ ) {
            par[n] = (*vec);
            n++;
         }

         Double_t effS = 0, effB = 0;
         this->MatchParsToCuts( par, &cutMin[0], &cutMax[0] );
         this->GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);

         for (Int_t ivar=0; ivar<fNvar; ivar++) {
            fCutMin[ivar][ibin-1] = cutMin[ivar]; // bin 1 stored in index 0
            fCutMax[ivar][ibin-1] = cutMax[ivar];
         }
      }
      delete[] cutMin;
      delete[] cutMax;

      // get elapsed time
      cout << "--- " << GetName() << ": GA: elapsed time: " << timer1.GetElapsedTime() 
           << endl;    

   }
   // --------------------------------------------------------------------------
   else {
      cerr << "--- " << GetName() << ": Error: unknown minization method: "
           << fFitMethod << " ==> abort" << endl;
      exit(1);    
   }

   // write weights and technical histos to file
   WriteWeightsToFile();
   WriteHistosToFile();
   delete fEffBvsSLocal;
   if (fBinaryTreeS) delete fBinaryTreeS;
   if (fBinaryTreeB) delete fBinaryTreeB;
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::ComputeEstimator( const std::vector<Double_t> & par )
{
   // returns estimator for "cut fitness" used by GA
   // there are two requirements:
   // 1) the signal efficiency must be equal to the required one in the 
   //    efficiency scan
   // 2) the background efficiency must be as small as possible
   // the requirement 1) has priority over 2)

   // caution: the npar gives the _free_ parameters
   // however: the "par" array contains all parameters

   // determine cuts
   Double_t effS = 0, effB = 0;
   this->MatchParsToCuts( par, &fTmpCutMin[0], &fTmpCutMax[0] );

   // retrieve signal and background efficiencies for given cut
   switch (fEffMethod) {
   case kUsePDFs:
      this->GetEffsfromPDFs( &fTmpCutMin[0], &fTmpCutMax[0], effS, effB );
      break;
   case kUseEventSelection:
      this->GetEffsfromSelection( &fTmpCutMin[0], &fTmpCutMax[0], effS, effB);
      break;
   default:
      this->GetEffsfromSelection( &fTmpCutMin[0], &fTmpCutMax[0], effS, effB);
   }

   // compute estimator
   const Double_t epsilon = 1.0e-06;
   Double_t eta;  
   if (fConstrainType == kConstrainEffS) {
      if (TMath::Abs(effS - fEffRef) > 0.001) eta = TMath::Abs(effB) + TMath::Abs(effS - fEffRef)/epsilon;
      else eta = TMath::Abs(effB);
   }
   else if (fConstrainType == kConstrainEffB) {
      eta = ( pow( (effB - fEffRef)/epsilon, 1 ) +
              pow( 1.0/((effS > 0) ? effS : epsilon), 2 ) );
   }
   else eta = 0;

   //   delete[] cutMin;
   //   delete[] cutMax;

   return eta;
}

//_______________________________________________________________________
void TMVA::MethodCuts::MatchParsToCuts( const std::vector<Double_t> & par, 
                                        Double_t* cutMin, Double_t* cutMax )
{
   // translates parameters into cuts
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      Int_t ipar = 2*ivar;
      cutMin[ivar] = ((*fRangeSign)[ivar] > 0) ? par[ipar] : par[ipar] - par[ipar+1];
      cutMax[ivar] = ((*fRangeSign)[ivar] > 0) ? par[ipar] + par[ipar+1] : par[ipar]; 
   }
}


//_______________________________________________________________________
void TMVA::MethodCuts::MatchCutsToPars( Double_t* par, 
                                        Double_t* cutMin, Double_t* cutMax )
{
   // translates cuts into parameters
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      Int_t ipar = 2*ivar;
      par[ipar]   = ((*fRangeSign)[ivar] > 0) ? cutMin[ivar] : cutMax[ivar];
      par[ipar+1] = cutMax[ivar] - cutMin[ivar];
   }
}

//_______________________________________________________________________
void TMVA::MethodCuts::GetEffsfromPDFs( Double_t* cutMin, Double_t* cutMax,
                                        Double_t& effS, Double_t& effB )
{
   // compute signal and background efficiencies from PDFs 
   // for given cut sample
   effS = 1.0;
   effB = 1.0;
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      effS *= (*fVarPdfS)[ivar]->GetIntegral( cutMin[ivar], cutMax[ivar] );
      effB *= (*fVarPdfB)[ivar]->GetIntegral( cutMin[ivar], cutMax[ivar] );
   }
}

//_______________________________________________________________________
void TMVA::MethodCuts::GetEffsfromSelection( Double_t* cutMin, Double_t* cutMax,
                                             Double_t& effS, Double_t& effB)
{
   // compute signal and background efficiencies from event counting 
   // for given cut sample
   Float_t nTotS = 0, nTotB = 0;
   Float_t nSelS = 0, nSelB = 0;  
  
   TMVA::Volume* volume = new TMVA::Volume( cutMin, cutMax, fNvar );
  
   nSelS = fBinaryTreeS->SearchVolume( volume );
   nSelB = fBinaryTreeB->SearchVolume( volume );
  
   nTotS = Float_t(fBinaryTreeS->GetSumOfWeights());
   nTotB = Float_t(fBinaryTreeB->GetSumOfWeights());
    
   delete volume;

   // sanity check
   if (nTotS == 0 && nTotB == 0) {
      cout << "--- " << GetName() 
           << ": fatal error in::ComputeEstimator: zero total number of events:"
           << " nTotS, nTotB: " << nTotS << " " << nTotB << " ***"
           << endl;
      exit(1);
   }

   // efficiencies
   if (nTotS == 0 ) {
      effS = 0;
      effB = nSelB/nTotB;
      cout << "--- " << GetName() 
           << ": Warning in ::ComputeEstimator: zero number of events signal Events:\n";
   }
   else if ( nTotB == 0) {
      effB = 0;
      effS = nSelS/nTotS;
      cout << "--- " << GetName() 
           << ": Warning in ::ComputeEstimator: zero number of events background Events:\n";
   }
   else {
      effS = nSelS/nTotS;
      effB = nSelB/nTotB;
   }  
}

//_______________________________________________________________________
void TMVA::MethodCuts::CreateVariablePDFs( void )
{
   // for PDF method: create efficiency reference histograms and PDFs

   // create list of histograms and PDFs
   fVarHistS        = new vector<TH1*>    ( fNvar );
   fVarHistB        = new vector<TH1*>    ( fNvar );
   fVarHistS_smooth = new vector<TH1*>    ( fNvar );
   fVarHistB_smooth = new vector<TH1*>    ( fNvar );
   fVarPdfS         = new vector<TMVA::PDF*>( fNvar );
   fVarPdfB         = new vector<TMVA::PDF*>( fNvar );

   Int_t nsmooth = 0;

   for (Int_t ivar=0; ivar<fNvar; ivar++) { 

      // ---- signal
      TString histTitle = (*fInputVars)[ivar] + " signal training";
      TString histName  = (*fInputVars)[ivar] + "_sig";
      TString drawOpt   = (*fInputVars)[ivar] + ">>h(";
      drawOpt += fNbins;
      drawOpt += ")";

      // selection
      fTrainingTree->Draw( drawOpt, "type==1", "goff" );
      (*fVarHistS)[ivar] = (TH1F*)gDirectory->Get("h");
      (*fVarHistS)[ivar]->SetName(histName);
      (*fVarHistS)[ivar]->SetTitle(histTitle);

      // make copy for smoothed histos
      (*fVarHistS_smooth)[ivar] = (TH1F*)(*fVarHistS)[ivar]->Clone();
      histTitle =  (*fInputVars)[ivar] + " signal training  smoothed ";
      histTitle += nsmooth;
      histTitle +=" times";
      histName =  (*fInputVars)[ivar] + "_sig_smooth";
      (*fVarHistS_smooth)[ivar]->SetName(histName);
      (*fVarHistS_smooth)[ivar]->SetTitle(histTitle);

      // smooth
      (*fVarHistS_smooth)[ivar]->Smooth(nsmooth);

      // ---- background
      histTitle = (*fInputVars)[ivar] + " background training";
      histName  = (*fInputVars)[ivar] + "_bgd";
      drawOpt   = (*fInputVars)[ivar] + ">>h(";
      drawOpt += fNbins;
      drawOpt += ")";

      fTrainingTree->Draw( drawOpt, "type==0", "goff" );
      (*fVarHistB)[ivar] = (TH1F*)gDirectory->Get("h");
      (*fVarHistB)[ivar]->SetName(histName);
      (*fVarHistB)[ivar]->SetTitle(histTitle);

      // make copy for smoothed histos
      (*fVarHistB_smooth)[ivar] = (TH1F*)(*fVarHistB)[ivar]->Clone();
      histTitle  = (*fInputVars)[ivar]+" background training  smoothed ";
      histTitle += nsmooth;
      histTitle +=" times";
      histName   = (*fInputVars)[ivar]+"_bgd_smooth";
      (*fVarHistB_smooth)[ivar]->SetName(histName);
      (*fVarHistB_smooth)[ivar]->SetTitle(histTitle);

      // smooth
      (*fVarHistB_smooth)[ivar]->Smooth(nsmooth);

      // create PDFs
      (*fVarPdfS)[ivar] = new TMVA::PDF( (*fVarHistS_smooth)[ivar], TMVA::PDF::kSpline2 );
      (*fVarPdfB)[ivar] = new TMVA::PDF( (*fVarHistB_smooth)[ivar], TMVA::PDF::kSpline2 );
   }                  
}

//_______________________________________________________________________
Bool_t TMVA::MethodCuts::SanityChecks( void )
{
   // basic checks to ensure that assumptions on variable order are satisfied
   Bool_t        isOK = kTRUE;

   TObjArrayIter branchIter( fTrainingTree->GetListOfBranches(), kIterForward );
   TBranch*      branch = 0;
   Int_t         ivar   = -1;
   while ((branch = (TBranch*)branchIter.Next()) != 0) {
      TString branchName = branch->GetName();

      if (branchName != "type") {

         // determine mean and rms to obtain appropriate starting values
         ivar++;
         if ((*fInputVars)[ivar] != branchName) {
            cout << "Error in: " << GetName() << "::SanityChecks: mismatch in variables ==> abort"
                 << endl;
            isOK = kFALSE;
         }
      }
   }  

   return isOK;
}

//_______________________________________________________________________
void  TMVA::MethodCuts::WriteWeightsToFile( void )
{
   // write cuts to file

   // though we could write the root effBvsS histogram directly, we
   // prefer here to put everything into a human-readable form  
   TString fname = GetWeightFileName();
   cout << "--- " << GetName() << ": creating weight file: " << fname << endl;
   ofstream fout( fname );
   if (!fout.good( )) { // file not found --> Error
      cout << "--- " << GetName() << ": Error in ::WriteWeightsToFile: "
           << "unable to open output  weight file: " << fname << endl;
      exit(1);
   }
   // write variable names and min/max
   // NOTE: the latter values are mandatory for the normalisation
   // in the reader application !!!
   fout << this->GetMethodName() <<endl;
   fout << "NVars= " << fNvar <<endl; 
   Int_t ivar;
   for (ivar=0; ivar<fNvar; ivar++) {
      TString var = (*fInputVars)[ivar];
      fout << var << "  " << GetXminNorm( var ) << "  " << GetXmaxNorm( var )
           << endl;
   }
  
   // first the dimensions
   fout << "OptimisationMethod " << "nRandCuts " << "nbins:" << endl;
   fout << ((fEffMethod == kUseEventSelection) ? "Fit-EventSelection" : 
            (fEffMethod == kUsePDFs) ? "Fit-PDF" : "Monte-Carlo") << "  " ;
   fout << fNRandCuts << "  ";
   fout << fNbins << endl;

   //  fout << endl;
   fout << "the optimised cuts for " << fNvar << " variables"  << endl;
   fout << "format: ibin(hist) effS effB cutMin[ivar=0] cutMax[ivar=0]"
        << " ... cutMin[ivar=n-1] cutMax[ivar=n-1]" << endl;
   Int_t ibin;
   for (ibin=0; ibin<fNbins; ibin++) {
      fout << setw(4) << ibin+1 << "  "    
           << setw(8)<< fEffBvsSLocal->GetBinCenter( ibin +1 ) << "  " 
           << setw(8)<< fEffBvsSLocal->GetBinContent( ibin +1 ) << "  ";  
      for (ivar=0; ivar<fNvar; ivar++)
         fout <<setw(10)<< fCutMin[ivar][ibin] << "  " << setw(10) << fCutMax[ivar][ibin] << "  ";
      fout << endl;
   }
}
  
//_______________________________________________________________________
void  TMVA::MethodCuts::ReadWeightsFromFile( void )
{
   // read cuts from file

   TString fname = GetWeightFileName();
   cout << "--- " << GetName() << ": reading weight file: " << fname << endl;
   ifstream fin( fname );
   if (!fin.good( )) { // file not found --> Error
      cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
           << "unable to open input file: " << fname << endl;
      exit(1);
   }

   // read variable names and min/max
   // NOTE: the latter values are mandatory for the normalisation
   // in the reader application !!!
   TString var, dummy;
   Double_t xmin, xmax;
   fin >> dummy;
   this->SetMethodName(dummy);
   fin >> dummy >> fNvar;
   Int_t ivar;
   for (ivar=0; ivar<fNvar; ivar++) {
      fin >> var >> xmin >> xmax;

      // sanity check
      if (var != (*fInputVars)[ivar]) {
         cout << "--- " << GetName() << ": Error while reading weight file; "
              << "unknown variable: " << var << " at position: " << ivar << ". "
              << "Expected variable: " << (*fInputVars)[ivar] << " ==> abort" 
              << endl;
         exit(1);
      }

      // set min/max
      this->SetXminNorm( ivar, xmin );
      this->SetXmaxNorm( ivar, xmax );
   }

   // first the dimensions
   fin >> dummy >> dummy >> dummy;
   fin >> dummy >> fNRandCuts >> fNbins;
   cout << "--- " << GetName() << ": Read cuts from "<< fNRandCuts << " MC events"
        << " in " << fNbins << " efficiency bins " << endl;
   fin >> dummy >> dummy >> dummy >> dummy >>fNvar>>dummy ;

   char buffer[200];
   fin.getline(buffer,200);
   fin.getline(buffer,200);

   // read histogram and cuts
   Int_t   ibin;
   Int_t   tmpbin;
   Float_t tmpeffS, tempeffB;
   for (ibin=0; ibin<fNbins; ibin++) {
      fin >> tmpbin >> tmpeffS >> tempeffB;

      if (ibin == 0       ) fEffSMin = tmpeffS;
      if (ibin == fNbins-1) fEffSMax = tmpeffS;

      for (ivar=0; ivar<fNvar; ivar++) {
         fin >> fCutMin[ivar][ibin] >> fCutMax[ivar][ibin];
      }
   }
}

//_______________________________________________________________________
void  TMVA::MethodCuts::WriteHistosToFile( void )
{
   // write histograms and PDFs (if exist) to file for monitoring purposes

   cout << "--- " << GetName() << ": write " << GetName() 
        << " special histos to file: " << fBaseDir->GetPath() << endl;
  
   fEffBvsSLocal->Write();

   // save reference histograms to file
   if (fEffMethod == kUsePDFs) {
      gDirectory->GetListOfKeys()->Print();
      fBaseDir->mkdir(GetName()+GetMethodName())->cd();  
      for (Int_t ivar=0; ivar<fNvar; ivar++) { 
         (*fVarHistS)[ivar]->Write();    
         (*fVarHistB)[ivar]->Write();
         (*fVarHistS_smooth)[ivar]->Write();    
         (*fVarHistB_smooth)[ivar]->Write();
         (*fVarPdfS)[ivar]->GetPDFHist()->Write();
         (*fVarPdfB)[ivar]->GetPDFHist()->Write();
      }
   }  
}

//_______________________________________________________________________
void TMVA::MethodCuts::TestInitLocal( TTree *theTree ) 
{
   // create binary trees (global member variables) for signal and background
   cout << "--- " << GetName() << ": called TestInitLocal " <<endl;

   fBinaryTreeS = new TMVA::BinarySearchTree();
   fBinaryTreeS->Fill( theTree, fInputVars, 1 );
   fBinaryTreeB = new TMVA::BinarySearchTree();
   fBinaryTreeB->Fill( theTree, fInputVars, 0 );
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::GetEfficiency( TString theString, TTree * /*theTree*/ )
{
   // - overloaded function to create background efficiency (rejection) versus
   //   signal efficiency plot (first call of this function)
   // - the function returns the signal efficiency at background efficiency
   //   indicated in theString
   //
   // "theString" must have two entries:
   // [0]: "Efficiency"
   // [1]: the value of background efficiency at which the signal efficiency 
   //      is to be returned
  
   // parse input string for required background efficiency
   TList* list  = TMVA::Tools::ParseFormatLine( theString );
   // sanity check
   if (list->GetSize() != 2) {
      cout << "--- " << GetName() << ": Error in::GetEfficiency: wrong number of arguments"
           << " in string: " << theString
           << " | required format, e.g., Efficiency:0.05" << endl;
      return -1;
   }

   // that will be the value of the efficiency retured (does not affect
   // the efficiency-vs-bkg plot which is done anyway.
   Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

   if (Verbose()) 
      cout << "--- " << GetName() << "::GetEfficiency <verbose>: compute eff(S) at eff(B) = " 
           << effBref << endl;

   // first round ? --> create histograms
   if ( fEffBvsS == NULL ||   fRejBvsS == NULL) {
      // there is no really good equivalent to the fEffS; fEffB (efficiency vs cutvalue)
      // for the "Cuts" method (unless we had only one cut). Maybe later I might add here
      // histograms for each of the cuts...but this would require also a change in the 
      // base class, and it is not really necessary, as we get exactly THIS info from the
      // "evaluateAllVariables" anyway.

      // now create efficiency curve: background versus signal
      //    if (NULL != fEffBvsS)fEffBvsS->Delete();
      //    if (NULL != fRejBvsS)fRejBvsS->Delete();
      if (NULL != fEffBvsS)delete fEffBvsS; 
      if (NULL != fRejBvsS)delete fRejBvsS; 
    
      fEffBvsS = new TH1F( fTestvar + "_effBvsS", fTestvar + "", fNbins, 0, 1 );
      fRejBvsS = new TH1F( fTestvar + "_rejBvsS", fTestvar + "", fNbins, 0, 1 );

      // use root finder

      // make the background-vs-signal efficiency plot
      Double_t* tmpCutMin = new Double_t[fNvar];
      Double_t* tmpCutMax = new Double_t[fNvar];
      for (Int_t bini=1; bini<=fNbins; bini++) {
         for (Int_t ivar=0; ivar <fNvar; ivar++){
            tmpCutMin[ivar] = fCutMin[ivar][bini-1];
            tmpCutMax[ivar] = fCutMax[ivar][bini-1];
         }
         // find cut value corresponding to a given signal efficiency
         Double_t effS, effB;
         this->GetEffsfromSelection( &tmpCutMin[0], &tmpCutMax[0], effS, effB);    

         // and fill histograms
         fEffBvsS->SetBinContent( bini, effB     );    
         fRejBvsS->SetBinContent( bini, 1.0-effB ); 
      }

      delete[] tmpCutMin;
      delete[] tmpCutMax;

      // create splines for histogram
      fGrapheffBvsS = new TGraph( fEffBvsS );
      fSpleffBvsS   = new TMVA::TSpline1( "effBvsS", fGrapheffBvsS );
   }

   // must exist...
   if (NULL == fSpleffBvsS) return 0.0;

   // now find signal efficiency that corresponds to required background efficiency
   Double_t effS, effB, effS_ = 0, effB_ = 0;
   Int_t    nbins_ = 1000;

   // loop over efficiency bins until the background eff. matches the requirement
   for (Int_t bini=1; bini<=nbins_; bini++) {
      // get corresponding signal and background efficiencies
      effS = (bini - 0.5)/Float_t(nbins_);
      effB = fSpleffBvsS->Eval( effS );

      // find signal efficiency that corresponds to required background efficiency
      if ((effB - effBref)*(effB_ - effBref) < 0) break;
      effS_ = effS;
      effB_ = effB;  
   }

   return 0.5*(effS + effS_);
}
