# Náš JNI most (whisper_jni.cpp) + whisper.cpp ako knižnica.
#
# DÔLEŽITÉ: whisper.cpp sa v tomto builde zostavuje ako SHARED (BUILD_SHARED_LIBS
# zapnuté AGP), takže jeho vlastný výstup je libwhisper.so. Náš JNI most preto
# MUSÍ mať iný názov (libwhisper_jni.so), inak si navzájom prepíšu výstup a
# Java_native symboly zmiznú.
cmake_minimum_required(VERSION 3.22.1)
project("odysseus-whisper")

# AGP Debug variant posiela -O0, čo robí whisper.cpp ~50x pomalším
# (5 s audio = 4 min). Vynútime optimalizácie aj v Debug builde.
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Len jadro: žiadne testy, examples, server, curl ani SDL.
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/whisper.cpp whispercpp-build)

add_library(odysseus_whisper_jni SHARED whisper_jni.cpp)
set_target_properties(odysseus_whisper_jni PROPERTIES OUTPUT_NAME "whisper_jni")
target_link_libraries(odysseus_whisper_jni PRIVATE whisper log)
