package eu.odysseus.diffusion

object GemmaLib {
    init {
        try { System.loadLibrary("smolvlm_jni") } catch (e: Exception) { android.util.Log.e("GemmaLib", "load failed: ${e.message}") }
    }
    external fun loadTextModel(modelPath: String): Long
    external fun generateText(handle: Long, prompt: String, maxTokens: Int): String?
    external fun releaseTextModel(handle: Long)
}
