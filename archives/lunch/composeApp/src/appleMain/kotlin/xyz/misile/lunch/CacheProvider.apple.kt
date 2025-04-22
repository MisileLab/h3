package xyz.misile.lunch

import kotlinx.serialization.json.Json
import platform.Foundation.NSUserDefaults

actual suspend inline fun <reified T> putDataRaw(context: Any?, key: String, data: T) {
    NSUserDefaults.standardUserDefaults().setObject(Json.encodeToString(data), key)
}

actual suspend inline fun <reified T> getDataRaw(context: Any?, key: String): T? {
    val value = NSUserDefaults.standardUserDefaults().stringForKey(key)
    return if (value == null) { null } else { Json.decodeFromString(value) }
}

actual suspend fun deleteData(context: Any?, key: String) {
    NSUserDefaults.standardUserDefaults().removeObjectForKey(key)
}

actual suspend fun clearData(context: Any?, key: String) {
    for (i in NSUserDefaults.standardUserDefaults().dictionaryRepresentation()) {
        val k = i.key
        if (k !is String) throw IllegalArgumentException("???")
        if (k.startsWith(key)) {
            NSUserDefaults.standardUserDefaults().removeObjectForKey(k)
        }
    }
}