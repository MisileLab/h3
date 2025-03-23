package xyz.misile.lunch

import kotlinx.datetime.Clock
import kotlinx.io.files.Path
import me.sujanpoudel.utils.paths.appCacheDirectory
import me.sujanpoudel.utils.paths.utils.div
import kotlin.math.abs

const val CACHE_DURATION = 3 * 60 * 60 * 1000

data class Cache<T> (
    val obj: T,
    val timestamp: Long
) {
    fun isExpired(other: Long): Boolean {
        return abs(timestamp - other) > CACHE_DURATION
    }
}

fun resolveCache(key: String?): String {
    if (key == null) return appCacheDirectory("xyz.misile.lunch").toString()
    return (appCacheDirectory("xyz.misile.lunch") / key).toString()
}

fun getCurrentTime(): Long {
    return Clock.System.now().epochSeconds
}

expect suspend fun clearData(context: Any?, key: String)
expect suspend fun deleteData(context: Any?, key: String)
expect suspend inline fun <reified T> putDataRaw(context: Any?, key: String, data: T)
expect suspend inline fun <reified T> getDataRaw(context: Any?, key: String): T?

suspend inline fun <reified T> putData(context: Any?, key: String, data: T) {
    return putDataRaw(context, key, Cache(data, getCurrentTime()))
}

suspend inline fun <reified T> getData(context: Any?, key: String): T? {
    val cache = getDataRaw<Cache<T>>(context, key) ?: return null
    if (cache.isExpired(getCurrentTime())) {
        deleteData(context, key)
        return null
    }
    return cache.obj
}
