plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "eu.odysseus.diffusion"
    compileSdk = 34

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
    }
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

    // Audio/video normalizácia: VÝHRADNE Min (LGPL). Full/GPL (x264/x265)
    // by vniesol GPL do APK — zakázané.
    // TODO(PIN): po overení LGPL-only artefaktu odkomentovať presnú verziu:
    // implementation("com.arthenica:ffmpeg-kit-min:6.0-2")

    // NCNN Vulkan (.so) + whisper.cpp JNI sa linkujú cez jniLibs, nie z Mavenu.
    // TODO(PIN): cmake externalNativeBuild po dodaní prebuilt .so.
}
