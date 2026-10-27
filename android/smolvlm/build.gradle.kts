plugins {
    id("com.android.library")
}

// llama.cpp + libmtmd live in their own CMake project on purpose: the app's
// CMakeLists already builds whisper.cpp with its own `ggml` target, and two
// CMake projects that both define `ggml` cannot share one build tree.
android {
    namespace = "eu.odysseus.smolvlm"
    compileSdk = 34
    ndkVersion = "27.3.13750724"

    defaultConfig {
        minSdk = 29
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared")
                cppFlags += listOf("-O3", "-fexceptions", "-frtti")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}
