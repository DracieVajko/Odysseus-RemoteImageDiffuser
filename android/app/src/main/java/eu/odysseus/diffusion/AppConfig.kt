package eu.odysseus.diffusion

/**
 * Runtime konfigurácia appky (nastavuje MainActivity pri START).
 * ApiServer ju číta pri rozhodovaní local vs remote.
 */
object AppConfig {
    @Volatile var mode: String = "AUTO"          // LOCAL | REMOTE | AUTO
    @Volatile var remoteBase: String = ""        // napr. http://192.168.2.110:8101
}
