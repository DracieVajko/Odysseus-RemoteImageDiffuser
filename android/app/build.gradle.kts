import java.io.File
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "eu.odysseus.diffusion"
    compileSdk = 34
    // PIN: NDK r27 (overené so whisper.cpp v1.9.5 + CMake 3.22.1).
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "eu.odysseus.diffusion"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            // NCNN Vulkan prebuilt .so sa dodávajú pre arm64; 32-bit ARM vypnutý.
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    packaging {
        resources {
            // Modely NIE v APK: váhy sa sťahujú cez ModelDownloader do app/files.
            excludes += setOf("**/*.bin", "**/*.param", "**/*.gguf", "META-INF/INDEX.LIST", "META-INF/io.netty.versions.properties", "META-INF/DEPENDENCIES")
        }
        // libwhisper.so sa balí z CMake buildu nižšie (arm64-v8a only).
    }

    // Natívny build len s NDK — bez neho čistý Kotlin build (WhisperLib stub).
    // Dôvod: NDK (~1 GB) nemá každý; povinný CMake by blokoval aj Ktor vývoj.
    val ndkPresent = findNdkDir() != null
    if (ndkPresent) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    } else {
        logger.warn("odysseus: NDK nenájdené → Kotlin-only build (libwhisper.so nebude v APK, audio ide stub cestou)")
    }
}

// Hľadá NDK v ANDROID_NDK_HOME, SDK/ndk/<ver> (sdk.dir z local.properties
// alebo ANDROID_HOME/ANDROID_SDK_ROOT). Vracia dir verzie alebo null.
fun findNdkDir(): File? {
    val explicit = System.getenv("ANDROID_NDK_HOME")?.let(::file)
    if (explicit != null && explicit.isDirectory) return explicit
    val props = Properties()
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { props.load(it) }
    val sdkRoot = (props.getProperty("sdk.dir") as String?)
        ?.let(::file)
        ?: System.getenv("ANDROID_HOME")?.let(::file)
        ?: System.getenv("ANDROID_SDK_ROOT")?.let(::file)
    val ndkRoot = sdkRoot?.resolve("ndk") ?: return null
    return ndkRoot.listFiles()?.firstOrNull { it.isDirectory }?.takeIf { it.resolve("build/cmake/android.toolchain.cmake").exists() }
        ?: ndkRoot.listFiles()?.firstOrNull { it.isDirectory }
}

dependencies {
    val ktor = "2.3.12"
    implementation("io.ktor:ktor-server-core:$ktor")
    implementation("io.ktor:ktor-server-netty:$ktor")
    implementation("io.ktor:ktor-server-content-negotiation:$ktor")
    implementation("io.ktor:ktor-server-status-pages:$ktor")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktor")
    implementation("io.ktor:ktor-client-core:$ktor")
    implementation("io.ktor:ktor-client-okhttp:$ktor")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")
    // ONNX Runtime (pre LCM-Dreamshaper INT8 ONNX inferenciu na mobile, V2).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")

    // Audio dekódovanie: Android MediaCodec (AudioDecoder.kt) — BEZ ffmpeg.
    // Ruling 2026-10-16: com.arthenica:ffmpeg-kit-min artefakty sú stiahnuté
    // z MavenCentral (404 aj na .pom, index je zastaraný) a JitPack vracia 401,
    // takže žiadny overiteľný Min/LGPL artefakt neexistuje. MediaCodec pokrýva
    // všetky kontraktové formáty (mp3/wav/m4a/webm/mp4/ogg/flac) s nulovým
    // licenčným rizikom; ffmpeg-kit sa preto vedome nezaraďuje.

    // whisper.cpp sa stavia zo zdrojov cez CMake nižšie (src/main/cpp) —
    // žiadne prebuilt .so, žiadny Maven. NCNN Vulkan príde rovnakou cestou (V2).
}
