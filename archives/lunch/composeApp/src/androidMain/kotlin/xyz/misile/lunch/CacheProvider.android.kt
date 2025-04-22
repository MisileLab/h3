package xyz.misile.lunch

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.Json

val Context.cache: DataStore<Preferences> by preferencesDataStore(name = "cache")

actual suspend inline fun <reified T>putDataRaw(context: Any?, key: String, data: T) {
    if (context !is Context) throw IllegalArgumentException("context must be a Context")
    context.cache.edit {
        it[stringPreferencesKey(key)] = Json.encodeToString(data)
    }
}

actual suspend inline fun <reified T>getDataRaw(context: Any?, key: String): T? {
    if (context !is Context) throw IllegalArgumentException("context must be a Context")
    val value = context.cache.data.first()[stringPreferencesKey(key)]
    return if (value == null) {
        null
    } else {
        Json.decodeFromString(value)
    }
}

actual suspend fun deleteData(context: Any?, key: String) {
    if (context !is Context) throw IllegalArgumentException("context must be a Context")
    context.cache.edit {
        it.remove(stringPreferencesKey(key))
    }
}

actual suspend fun clearData(context: Any?, key: String) {
    if (context !is Context) throw IllegalArgumentException("context must be a Context")
    val keys = context.cache.data.first().asMap().keys
    context.cache.edit {
        for (i in keys) {
            if (i.name.startsWith(key)) {it.remove(i)}
        }
    }
}