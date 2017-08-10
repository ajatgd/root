#ifndef PARSING
#define PARSING

#include "TString.h"
#include "TObject.h"
#include <vector>
#include <map>

#include "TString.h"

#include "TMVA/Tools.h"







namespace TMVA 
{
  namespace DNNParsing {  
  

 
  TString fetchValue (const std::map<TString, TString>& keyValueMap, TString key); 
     
  std::vector<double> fetchValue(const std::map<TString, TString> & keyValueMap, TString key, std::vector<double> defaultValue);             

  int fetchValue(const std::map<TString,TString>& keyValueMap, TString key, int defaultValue); 

  double fetchValue (const std::map<TString,TString>& keyValueMap, TString key, double defaultValue); 
 
  TString fetchValue (const std::map<TString,TString>& keyValueMap, TString key, TString defaultValue); 
      
  bool fetchValue (const std::map<TString,TString>& keyValueMap, TString key, bool defaultValue); 
      

  
  }

}


#endif
