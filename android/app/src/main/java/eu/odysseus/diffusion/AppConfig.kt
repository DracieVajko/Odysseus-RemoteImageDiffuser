package eu.odysseus.diffusion

/**
 * Runtime configuration for the app.
 * Everything runs locally on the phone; there is no remote/PC mode.
 */
object AppConfig {
    @Volatile var serverPort: Int = 8101

    /** "auto" | "cpu" | "gpu" — selects ONNX execution providers. */
    @Volatile var acceleration: String = "auto"
}
