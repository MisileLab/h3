package xyz.misile.lunch

import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import java.io.File

actual suspend inline fun <reified T>putDataRaw(context: Any?, key: String, data: T) {
    val f = File(resolveCache(null))
    if (!f.exists()) f.mkdirs()
    File(resolveCache(key)).writeText(Json.encodeToString(data))
}

actual suspend inline fun <reified T>getDataRaw(context: Any?, key: String): T? {
    val f = File(resolveCache(null))
    if (!f.exists()) {
        f.mkdirs()
        return null
    }
    return try {
        Json.decodeFromString<T>(File(resolveCache(key)).readText())
    } catch (e: SerializationException) {
        null
    }
}

actual suspend fun deleteData(context: Any?, key: String) {
    File(resolveCache(key)).delete()
}

actual suspend fun clearData(context: Any?, key: String) {
    for (i in File(resolveCache(null)).listFiles { file -> file.isFile && file.name.startsWith(key) }!!) {
        i.delete()
    }
}