# Our JNI bridge (whisper_jni.cpp) + whisper.cpp as a library.
#
# IMPORTANT: in this build whisper.cpp is compiled as SHARED (BUILD_SHARED_LIBS
# is enabled by AGP), so its own output is libwhisper.so. Our JNI bridge must
# therefore use a different name (libwhisper_jni.so); otherwise the two
# overwrite each other's output and the Java_native symbols disappear.
cmake_minimum_required(VERSION 3.22.1)
project("odysseus-whisper")

# The AGP Debug variant passes -O0, which makes whisper.cpp ~50x slower
# (5 s of audio = 4 min). Force optimizations in the Debug build too.
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Core only: no tests, examples, server, curl or SDL.
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/whisper.cpp whispercpp-build)

add_library(odysseus_whisper_jni SHARED whisper_jni.cpp)
set_target_properties(odysseus_whisper_jni PROPERTIES OUTPUT_NAME "whisper_jni")
target_link_libraries(odysseus_whisper_jni PRIVATE whisper log)

# --- Real-ESRGAN upscale (ncnn Vulkan, prebuilt static libs) ---
# The ncnn Android Vulkan package is downloaded by tools/fetch_ncnn.ps1 into
# third_party/ncnn (gitignored). Without it the JNI lib is simply not built and
# the Kotlin engine returns 503.
set(NCNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn-shared/ncnn-20261020-android-vulkan-shared/${ANDROID_ABI}")
if(EXISTS "${NCNN_ROOT}/lib/libncnn.so")
    set(ncnn_DIR "${NCNN_ROOT}/lib/cmake/ncnn")
    find_package(ncnn REQUIRED)
    add_library(odysseus_realesrgan_jni SHARED realesrgan_jni.cpp)
    set_target_properties(odysseus_realesrgan_jni PROPERTIES OUTPUT_NAME "realesrgan_jni")
    target_link_libraries(odysseus_realesrgan_jni PRIVATE ncnn)
else()
    message(STATUS "ncnn not found for ${ANDROID_ABI}; skipping realesrgan_jni")
endif()
