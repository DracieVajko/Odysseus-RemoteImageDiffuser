# Náš JNI most (whisper_jni.cpp) + whisper.cpp ako statická knižnica.
#
# whisper.cpp zdrojáky NIE sú v gite (android/app/src/main/cpp/third_party/,
# klonuje fetch skript) — tu len tenký most nad stabilným C API, ktorý
# plne rozumieme: init_from_file / full / segments / free.
cmake_minimum_required(VERSION 3.22.1)
project("odysseus-whisper")

# Len jadro: žiadne testy, examples, server, curl ani SDL.
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/whisper.cpp whispercpp-build)

# Vlastný shared objekt MUSÍ mať výstup libwhisper.so
# (WhisperLib.kt volá System.loadLibrary("whisper")), ale cieľ sa nemôže
# volať "whisper" — to meno už má statická knižnica whisper.cpp.
add_library(odysseus_whisper_jni SHARED whisper_jni.cpp)
set_target_properties(odysseus_whisper_jni PROPERTIES OUTPUT_NAME "whisper")
target_link_libraries(odysseus_whisper_jni PRIVATE whisper log)
