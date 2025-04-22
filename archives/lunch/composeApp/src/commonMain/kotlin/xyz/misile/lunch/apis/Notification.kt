package xyz.misile.lunch.apis

import io.ktor.client.call.body
import io.ktor.client.request.get
import kotlinx.serialization.Serializable

@Serializable
data class Notification(
    val id: String,
    val title: String,
    val date: String,
    val content: String
)

suspend fun getNotifications(): Array<Notification> {
    return client.get("/notifications").body()
}
