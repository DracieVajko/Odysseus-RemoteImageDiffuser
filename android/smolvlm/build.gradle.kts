import java.util.Properties

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

    // CI (and any machine without the NDK) still builds the module, just
    // without the native library; the engine then reports "not bundled".
    if (findNdkDir() != null) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    }
}

fun findNdkDir(): java.io.File? {
    val explicit = System.getenv("ANDROID_NDK_HOME")?.let(::file)
    if (explicit != null && explicit.isDirectory) return explicit
    val props = Properties()
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { props.load(it) }
    val sdkRoot = (props.getProperty("sdk.dir") as String?)?.let(::file)
        ?: System.getenv("ANDROID_HOME")?.let(::file)
        ?: System.getenv("ANDROID_SDK_ROOT")?.let(::file)
    val ndkRoot = sdkRoot?.resolve("ndk") ?: return null
    return ndkRoot.listFiles()?.firstOrNull { it.isDirectory }
        ?.takeIf { it.resolve("build/cmake/android.toolchain.cmake").exists() }
}
