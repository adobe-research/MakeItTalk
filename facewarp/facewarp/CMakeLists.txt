cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME "facewarp")

add_subdirectory(src)
add_subdirectory(third_party)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})