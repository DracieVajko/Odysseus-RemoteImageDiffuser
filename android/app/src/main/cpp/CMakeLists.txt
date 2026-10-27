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

# --- MP3 encoding (libmp3lame, LGPL, statically linked into our JNI) ---
# Android has no MP3 encoder in MediaCodec. LAME is ~1 MB of C and solves that
# without bundling ffmpeg. Sources come from tools/fetch_third_party.ps1.
# LAME lives next to llama.cpp in android/third_party (gitignored, fetched by
# tools/fetch_third_party.ps1): cpp -> main -> src -> app -> android.
set(LAME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/lame")
if(EXISTS "${LAME_DIR}/libmp3lame/lame.c")
    file(GLOB LAME_SRC "${LAME_DIR}/libmp3lame/*.c")
    add_library(mp3lame STATIC ${LAME_SRC})
    target_include_directories(mp3lame PUBLIC
            "${LAME_DIR}/include"
            "${LAME_DIR}/libmp3lame")
    target_include_directories(mp3lame PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lame_config")
    # -include: LAME includes machine.h before config.h in some TUs, and
    # that header switches memcpy to bcopy unless STDC_HEADERS is already set.
    target_compile_options(mp3lame PRIVATE -O3 -w
            -include "${CMAKE_CURRENT_SOURCE_DIR}/lame_config/config.h")

    add_library(odysseus_mp3_jni SHARED mp3_jni.cpp)
    set_target_properties(odysseus_mp3_jni PROPERTIES OUTPUT_NAME "mp3_jni")
    target_link_libraries(odysseus_mp3_jni PRIVATE mp3lame log)
else()
    message(STATUS "libmp3lame not found; MP3 presets stay unsupported")
endif()
