# 1. Minimum CMake version (standard requirement)
cmake_minimum_required(VERSION 3.10)

# 2. Project name and language (CXX stands for C++)
project(SmartUAVScout CXX)

# Set C++17 standard (best for modern OpenCV)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. CMake magic: Find OpenCV automatically
# REQUIRED means that if CMake cannot find OpenCV, it will throw an error and stop
find_package(OpenCV REQUIRED)

# Print a message to the console to verify OpenCV was found
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 4. Tell CMake which files to use to build our application (.exe)
# SmartUAVScout is the name of our future executable
# src/main.cpp is our source code
add_executable(SmartUAVScout src/main.cpp)

# 5. Include OpenCV header files (.h) to our executable
target_include_directories(SmartUAVScout PRIVATE ${OpenCV_INCLUDE_DIRS})

# 6. Link OpenCV libraries (.lib) to our executable
target_link_libraries(SmartUAVScout PRIVATE ${OpenCV_LIBS})

# Optional: do the same for the video file (if it's in the root directory)
# add_custom_command(TARGET SmartUAVScout POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${CMAKE_SOURCE_DIR}/video.mp4"
#     "$<TARGET_FILE_DIR:SmartUAVScout>/video.mp4"
# )
# ==========================================
# 7. ONNX Runtime Integration (GPU Version)
# ==========================================

# Define the path to our extracted ONNX Runtime library
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

# 7.1 Include the "dictionary" (Headers) so the compiler knows the ORT API
target_include_directories(SmartUAVScout PRIVATE "${ORT_DIR}/include")

# 7.2 Link the static library (.lib) for the build phase
target_link_libraries(SmartUAVScout PRIVATE "${ORT_DIR}/lib/onnxruntime.lib")

# 7.3 Copy dynamic libraries (.dll) next to our .exe AFTER building to prevent runtime crashes
add_custom_command(TARGET SmartUAVScout POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ORT_DIR}/lib/onnxruntime.dll"
    "$<TARGET_FILE_DIR:SmartUAVScout>/onnxruntime.dll"
    COMMENT "Copying onnxruntime.dll to output directory..."
)

add_custom_command(TARGET SmartUAVScout POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ORT_DIR}/lib/onnxruntime_providers_shared.dll"
    "$<TARGET_FILE_DIR:SmartUAVScout>/onnxruntime_providers_shared.dll"
    COMMENT "Copying onnxruntime_providers_shared.dll..."
)

add_custom_command(TARGET SmartUAVScout POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ORT_DIR}/lib/onnxruntime_providers_cuda.dll"
    "$<TARGET_FILE_DIR:SmartUAVScout>/onnxruntime_providers_cuda.dll"
    COMMENT "Copying onnxruntime_providers_cuda.dll (for RTX 3050!)..."
)

# Copy the model from the global models folder to the executable directory AFTER build
add_custom_command(TARGET SmartUAVScout POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/../models/deployment/smart_scout_v1.onnx"
    "$<TARGET_FILE_DIR:SmartUAVScout>/smart_scout_v1.onnx"
    COMMENT "Copying smart_scout_v1.onnx to the output directory..."
)

# Copy video.mp4 next to the .exe AFTER every successful build
add_custom_command(TARGET SmartUAVScout POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/video.mp4"
    "$<TARGET_FILE_DIR:SmartUAVScout>/video.mp4"
    COMMENT "Copying video.mp4 to the output directory..."
)

# ==========================================
# 8. AUTO-COPY cuDNN DLLs
# ==========================================

set(CUDNN_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn/bin")
file(GLOB CUDNN_DLLS "${CUDNN_BIN_DIR}/*.dll")

foreach(DLL_FILE ${CUDNN_DLLS})
    add_custom_command(TARGET SmartUAVScout POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLL_FILE}"
        $<TARGET_FILE_DIR:SmartUAVScout>
        COMMENT "Auto-copying cuDNN dependency: ${DLL_FILE}"
    )
endforeach()