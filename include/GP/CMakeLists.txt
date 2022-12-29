cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

add_library(
    GP
    INTERFACE)

target_include_directories(
    GP
    INTERFACE .
)
target_compile_options(GP
    INTERFACE
   -mavx -mavx2
)
find_package(OpenMP)
message(CHECK_START "Finding OpenMP...")
if(OpenMP_CXX_FOUND)
    message(CHECK_PASS "Found OpenMP.")
    target_link_libraries(GP INTERFACE OpenMP::OpenMP_CXX)
else()
    message(CHECK_FAIL "OpenMP not found.")
endif()