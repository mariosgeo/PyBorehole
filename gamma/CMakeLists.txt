cmake_minimum_required(VERSION 3.16)
project(gammalog CXX)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(gammalog gammalog.cc)
target_compile_features(gammalog PRIVATE cxx_std_17)
target_link_libraries(gammalog ${Geant4_LIBRARIES})
