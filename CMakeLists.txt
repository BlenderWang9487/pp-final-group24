cmake_minimum_required(VERSION 3.16)
project(pp_final LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(include/GP)

add_executable(app
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(app
    PRIVATE GP
)
set_target_properties(app
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)