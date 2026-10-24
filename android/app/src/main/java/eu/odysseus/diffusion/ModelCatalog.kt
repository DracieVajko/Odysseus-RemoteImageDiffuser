package eu.odysseus.diffusion

/**
* Model catalog — the single source of truth for Android.
 *
* Rule: every URL is verified (HTTP 200 + Content-Length) and every model
* has an exact file list. Directory-style HF repos (e.g. LCM-Dreamshaper
* ONNX) are downloaded file by file, not as a "directory URL" — that was the
* reason the DreamShaper download failed before.
 *
* `engine` says what can actually run the model:
*  - whisper.cpp  → local transcription (JNI, works)
*  - onnxruntime  → local inference (engine in progress; weights downloadable)
*  - vulkan-ncnn  → Real-ESRGAN (weights downloadable; engine in progress)
*  - llama.cpp    → VLM runtime not bundled yet (experimental on-device)
 */
data class ModelFile(
    val path: String,
    val url: String,
    val sizeMb: Long,
    val sha256: String = "",
)

data class ModelAsset(
    val id: String,
    val kind: String, // audio | image | upscale | vlm | custom
    val label: String,
    val files: List<ModelFile>,
    val engine: String = "",
    val note: String = "",
    val custom: Boolean = false,
) {
    val sizeMb: Long get() = files.sumOf { it.sizeMb }

/** First URL — backward compatibility with the UI and API (`has_source`). */
    val url: String get() = files.firstOrNull()?.url ?: ""

/** SHA-256 of the first file (legacy API field). */
    val sha256: String get() = files.firstOrNull()?.sha256 ?: ""

    val hasSource: Boolean get() = files.isNotEmpty()

/** True if the model can run directly on the phone. */
    val runsOnDevice: Boolean get() = engine != "llama.cpp"
}

object ModelCatalog {
    private const val WHISPER = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

    private fun ggml(name: String, sizeMb: Long, sha: String = ""): ModelFile =
        ModelFile("ggml-$name.bin", "$WHISPER/ggml-$name.bin", sizeMb, sha)

/** Whisper (whisper.cpp GGML) — local transcription. */
    private val audio: List<ModelAsset> = listOf(
        ModelAsset(
            id = "whisper-large-v3-turbo-q5_0", kind = "audio", engine = "whisper.cpp",
            label = "Whisper Large v3 Turbo q5_0 — 547 MB, SK/CZ, fastest large model",
            files = listOf(ggml("large-v3-turbo-q5_0", 547)),
            note = "Default audio model. ~2.5x realtime on SD 8 Gen 2 (~2 h for 45 min).",
        ),
    )

    /**
* LCM-Dreamshaper v7 INT8 ONNX (the repo is a directory, hence the explicit
* file list). The UNet graph is small; weights live in an external .data file —
* both must be downloaded.
     */
    private val image: List<ModelAsset> = listOf(
        ModelAsset(
            id = "lcm-dreamshaper-int8-onnx", kind = "image", engine = "onnxruntime",
            label = "LCM-Dreamshaper v7 INT8 ONNX — 1.6 GB, 4 steps",
            files = listOf(
                ModelFile("unet/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model.onnx", 3),
                ModelFile("unet/model_int8.onnx.data",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model_int8.onnx.data", 821),
                ModelFile("text_encoder/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/text_encoder/model.onnx", 470),
                ModelFile("vae_decoder/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/vae_decoder/model.onnx", 189),
                ModelFile("vae_encoder/model.onnx",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/vae_encoder/model.onnx", 130),
                ModelFile("tokenizer/vocab.json",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/tokenizer/vocab.json", 1),
                ModelFile("tokenizer/merges.txt",
                    "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/tokenizer/merges.txt", 1),
            ),
            note = "JSON graph + external weights; engine in progress. Use PC (:8101) for NVIDIA.",
        ),
    )

    private val upscale: List<ModelAsset> = listOf(
        ModelAsset(
            id = "realesrgan-x4plus-ncnn", kind = "upscale", engine = "vulkan-ncnn",
            label = "Real-ESRGAN x4plus NCNN — 45 MB",
            files = listOf(
                ModelFile("realesrgan-x4plus-ncnn.zip",
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip", 45),
            ),
            note = "Zip contains .param/.bin weights; auto-extracted.",
        ),
    )

/** VLM: runtime is llama.cpp (libmtmd) — not bundled yet; experimental. */
    private val vlm: List<ModelAsset> = listOf(
        ModelAsset(
            id = "smolvlm-500m-gguf", kind = "vlm", engine = "llama.cpp",
            label = "SmolVLM 500M Instruct Q8_0 + vision mmproj — 520 MB",
            files = listOf(
                ModelFile("SmolVLM-500M-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf", 417),
                ModelFile("mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", 104),
            ),
            note = "Mobile: caption/chat runs via PC in Auto/Remote mode.",
        ),
        ModelAsset(
            id = "smolvlm2-500m-gguf", kind = "vlm", engine = "llama.cpp",
            label = "SmolVLM2 500M Video Instruct Q8_0 + mmproj — 520 MB",
            files = listOf(
                ModelFile("SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf", 417),
                ModelFile("mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf", 104),
            ),
            note = "Mobile: caption/chat runs via PC in Auto/Remote mode.",
        ),
    )

    val assets: List<ModelAsset> = audio + image + upscale + vlm

    val kinds: List<String> = listOf("audio", "image", "upscale", "vlm", "custom")

    fun kindLabel(kind: String): String = when (kind) {
        "audio" -> "Transcription (Whisper)"
        "image" -> "Image generation"
        "upscale" -> "Upscale"
        "vlm" -> "Vision-language (VLM)"
        "custom" -> "Custom (own URL)"
        else -> kind
    }

    fun find(id: String): ModelAsset? = assets.firstOrNull { it.id == id }

    fun forImageModel(registryId: String): ModelAsset =
        assets.firstOrNull { it.kind == "image" && it.id == registryId }
            ?: assets.first { it.kind == "image" }

    fun forKind(kind: String): List<ModelAsset> = assets.filter { it.kind == kind }
}
