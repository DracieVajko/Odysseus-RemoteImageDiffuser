import java.io.File
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    // Keep serialization plugin for @Serializable annotations
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.24"
}

android {
    namespace = "eu.odysseus.diffusion"
    compileSdk = 34
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "eu.odysseus.diffusion"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        ndk {
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
            excludes += setOf("**/*.bin", "**/*.param", "**/*.gguf", "META-INF/INDEX.LIST", "META-INF/io.netty.versions.properties", "META-INF/DEPENDENCIES")
        }
    }

    val ndkPresent = findNdkDir() != null
    if (ndkPresent) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    } else {
        logger.warn("odysseus: NDK not found → Kotlin-only build (libwhisper.so will not be in the APK, audio uses the stub)")
    }
}

fun findNdkDir(): File? {
    val explicit = System.getenv("ANDROID_NDK_HOME")?.let(::file)
    if (explicit != null && explicit.isDirectory) return explicit
    val props = Properties()
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { props.load(it) }
    val sdkRoot = (props.getProperty("sdk.dir") as String?)?.let(::file)
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
    // Use Gson for JSON parsing to avoid kotlinx.serialization version conflicts
    implementation("io.ktor:ktor-serialization-gson:$ktor")
    implementation("io.ktor:ktor-client-core:$ktor")
    implementation("io.ktor:ktor-client-okhttp:$ktor")
    implementation("com.google.code.gson:gson:2.10.1")
    // Keep kotlinx-serialization for @Serializable annotations (needed for data classes)
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.5.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.3")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}