find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME}
    facewarp.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CMAKE_CXX_STANDARD 11
    CMAKE_CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${OPENGL_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    glfw
    ${OPENGL_gl_LIBRARY}
    stb
)

if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
endif()