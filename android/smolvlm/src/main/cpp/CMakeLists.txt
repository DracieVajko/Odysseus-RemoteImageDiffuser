# SmolVLM (llama.cpp + libmtmd) as a standalone Native build.
#
# Kept out of app/src/main/cpp on purpose: that project already builds
# whisper.cpp, which brings its own `ggml` target. Two CMake projects both
# defining `ggml` in one build tree silently share the first one and the second
# fails against newer headers. This project builds llama.cpp's ggml statically
# and links it into libsmolvlm_jni.so, so no extra .so names appear in the APK.
cmake_minimum_required(VERSION 3.22.1)
project(smolvlm LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Release-quality code even in the debug variant (llama.cpp at -O0 is unusable).
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp")
if(NOT EXISTS "${LLAMA_DIR}/include/llama.h")
    message(WARNING "llama.cpp missing (run android/tools/fetch_third_party.ps1); smolvlm_jni is not built")
    return()
endif()
message(STATUS "smolvlm: using llama.cpp at ${LLAMA_DIR}")

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_SHARED OFF CACHE BOOL "" FORCE)
set(LLAMA_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)

add_subdirectory(${LLAMA_DIR} llama-build)

set(MTMD_DIR "${LLAMA_DIR}/tools/mtmd")
find_package(Threads REQUIRED)
file(GLOB MTMD_SRC "${MTMD_DIR}/*.cpp" "${MTMD_DIR}/models/*.cpp")
list(REMOVE_ITEM MTMD_SRC "${MTMD_DIR}/mtmd-cli.cpp")

add_library(smolvlm_jni SHARED
        smolvlm_jni.cpp
        ${LLAMA_DIR}/vendor/hash/hash.cpp
        ${LLAMA_DIR}/vendor/hash/sha1/sha1.c
        ${LLAMA_DIR}/vendor/hash/sha256/sha256.c
        ${LLAMA_DIR}/vendor/hash/xxhash/xxhash.c
        ${MTMD_SRC})

# sha1.c is C++ (it uses a namespace to avoid a clash with boringssl);
# sha256.c and xxhash.c stay C, exactly like llama.cpp's own vendor target.
set_source_files_properties(
        ${LLAMA_DIR}/vendor/hash/sha1/sha1.c
        PROPERTIES LANGUAGE CXX)

target_include_directories(smolvlm_jni PRIVATE
        "${MTMD_DIR}"
        "${LLAMA_DIR}/include"
        "${LLAMA_DIR}/ggml/include"
        "${LLAMA_DIR}/vendor"
        "${LLAMA_DIR}/vendor/stb"
        "${LLAMA_DIR}/vendor/sheredom"
        "${LLAMA_DIR}/vendor/miniaudio"
        "${LLAMA_DIR}/vendor/hash")

target_link_libraries(smolvlm_jni PRIVATE llama ggml Threads::Threads android log jnigraphics)
set_target_properties(smolvlm_jni PROPERTIES OUTPUT_NAME "smolvlm_jni")
