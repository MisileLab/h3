package xyz.misile.lunch.apis

import io.ktor.client.plugins.ClientRequestException
import io.ktor.client.request.delete
import io.ktor.client.request.get
import io.ktor.client.request.parameter
import io.ktor.client.request.post
import io.ktor.client.request.put
import io.ktor.http.HttpStatusCode

suspend fun addFcmToken(
    token: String,
    time: String,
    schoolCode: String,
    regionCode: String
) {
    client.post("/fcm") {
        parameter("token", token)
        parameter("time", time)
        parameter("schoolCode", schoolCode)
        parameter("regionCode", regionCode)
    }
}

suspend fun removeFcmToken(token: String) {
    client.delete("/fcm") {
        parameter("token", token)
    }
}

suspend fun checkFcmToken(token: String): Boolean {
    try {
        client.get("/fcm") {
            parameter("token", token)
        }
        return true
    } catch (e: ClientRequestException) {
        if (e.response.status == HttpStatusCode.NotFound) {
            return false
        }
        throw e
    }
}

suspend fun editFcmTime(
    token: String,
    time: String,
    schoolCode: String,
    regionCode: String
) {
    client.put("/fcm") {
        parameter("token", token)
        parameter("time", time)
        parameter("schoolCode", schoolCode)
        parameter("regionCode", regionCode)
    }
}
