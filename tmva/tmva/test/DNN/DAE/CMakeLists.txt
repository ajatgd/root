############################################################################
# CMakeLists.txt file for building TMVA/DNN/CNN tests.
# @author Akshay Vashistha
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

ROOT_EXECUTABLE(testCorruptionLayer TestCorruptionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-CorruptionLayer COMMAND testCorruptionLayer)

ROOT_EXECUTABLE(testCompressionLayer TestCompressionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-CompressionLayer COMMAND testCompressionLayer)

ROOT_EXECUTABLE(testReconstructionLayer TestReconstructionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-ReconstructionLayer COMMAND testReconstructionLayer)

ROOT_EXECUTABLE(testLogisticRegressionLayer TestLogisticRegressionLayer.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-LogisticRegressionLayer COMMAND testLogisticRegressionLayer)

ROOT_EXECUTABLE(testNetPreTrain TestNetPreTrain.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-NetPreTrain COMMAND testNetPreTrain)

ROOT_EXECUTABLE(testNetFineTune TestNetFineTune.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-DAE-NetFineTune COMMAND testNetFineTune)

#--- CPU tests. ----------------------------
if (BLAS_FOUND AND imt)

include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

ROOT_EXECUTABLE(testCorruptionLayerCpu TestCorruptionLayerCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-CorruptionLayer-CPU COMMAND testCorruptionLayerCpu)

ROOT_EXECUTABLE(testCompressionLayerCpu TestCompressionLayerCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-CompressionLayer-CPU COMMAND testCompressionLayerCpu)

ROOT_EXECUTABLE(testLogisticRegressionLayerCpu TestLogisticRegressionLayerCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-LogisticRegressionLayer-CPU COMMAND testLogisticRegressionLayerCpu)

ROOT_EXECUTABLE(testNetFineTuneCpu TestNetFineTuneCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-NetFineTune-CPU COMMAND testNetFineTuneCpu)

ROOT_EXECUTABLE(testNetPreTrainCpu TestNetPreTrainCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-NetPreTrain-CPU COMMAND testNetPreTrainCpu)

ROOT_EXECUTABLE(testReconstructionLayerCpu TestReconstructionLayerCpu.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-DNN-CNN-ReconstructionLayer-CPU COMMAND testReconstructionLayerCpu)

endif (BLAS_FOUND AND imt)
