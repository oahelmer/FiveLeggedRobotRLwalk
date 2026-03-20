cmake_minimum_required(VERSION 3.10)
project(FiveLeggedRobot)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Dependencies
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# GLM setup
find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "GLM not found via find_package, searching in common paths...")
    find_path(GLM_INCLUDE_DIRS "glm/glm.hpp")
endif()

# Define ONNX Runtime paths from your src directory
set(ONNX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/onnxruntime-linux-x64-gpu-1.24.4)
set(ONNX_INCLUDE_DIRS ${ONNX_ROOT}/include)
set(ONNX_LIBRARIES ${ONNX_ROOT}/lib/libonnxruntime.so)

# Define source files including the new RL controller
set(SOURCES 
    main.cpp
    src/constants.cpp
    src/robot-from-above.cpp
    src/RLwalking.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${ONNX_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    OpenGL::GL 
    GLEW::GLEW 
    glfw
    ${ONNX_LIBRARIES}
)

# Copy headers for clarity in IDEs
set(HEADERS
    src/constants.h
    src/robot-from-above.h
    src/RLwalking.h
)
target_sources(${PROJECT_NAME} PUBLIC ${HEADERS})

# Ensure the .so file is found at runtime
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${ONNX_ROOT}/lib")
set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

# Copy the RL model to the build directory so the executable can find it
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/rl_policy.onnx 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})