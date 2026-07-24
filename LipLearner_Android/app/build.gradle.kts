plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.rkmtlab.liplearner"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.rkmtlab.liplearner"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        // Real phones are arm64; drop other ABIs to shrink the APK (the ONNX models dominate size).
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
    // The ONNX / MediaPipe model assets ship uncompressed so they can be mmap'd at runtime.
    androidResources {
        noCompress += listOf("onnx", "task", "tflite")
    }
    packaging {
        resources.excludes += "META-INF/*"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.activity:activity-ktx:1.9.1")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // CameraX
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // MediaPipe Face Landmarker (face mesh incl. lips) — replaces iOS Vision
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    // ONNX Runtime Mobile — replaces CoreML
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
}
