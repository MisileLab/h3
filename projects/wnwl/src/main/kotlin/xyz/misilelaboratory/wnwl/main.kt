package xyz.misilelaboratory.wnwl

import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.ratelimit.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import org.bukkit.Bukkit
import org.bukkit.OfflinePlayer
import org.bukkit.plugin.java.JavaPlugin
import java.util.*
import kotlin.collections.HashMap
import kotlin.time.Duration.Companion.seconds

@Serializable
data class MojangData(val id: String, val name: String)

@Suppress("unused")
class Wnwl: JavaPlugin() {

    private val pretendPlayers = HashMap<String, OfflinePlayer>()

    override fun onEnable() {
        logger.info("Enabled")
        embeddedServer(Netty, port = server.port + 1) {
            install(RateLimit) {
                global {
                    rateLimiter(limit = 3, refillPeriod = 60.seconds)
                }
            }
            routing {
                post("add/{name}") {
                    val name = call.parameters["name"]
                    if (name == null) {
                        call.respond(HttpStatusCode.BadRequest)
                    } else {
                        kotlin.run {
                            val client = HttpClient(CIO) {}
                            val res = client.get("https://api.mojang.com/users/profiles/minecraft/$name")
                            if (res.status != HttpStatusCode.OK) {
                                call.respond(HttpStatusCode.NotFound)
                                client.close()
                            } else {
                                pretendPlayers[name] = Bukkit.getOfflinePlayer(UUID.fromString(Json.decodeFromString<MojangData>(res.bodyAsText()).id))
                                // add event
                            }
                        }
                    }
                }
            }
        }.start(wait = true)
    }
}