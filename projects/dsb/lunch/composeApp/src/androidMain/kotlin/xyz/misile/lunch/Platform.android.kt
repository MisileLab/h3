package xyz.misile.lunch

import android.os.Build
import io.ktor.client.HttpClient
import io.ktor.client.HttpClientConfig

class AndroidPlatform : Platform {
    override val name: String = "Android ${Build.VERSION.SDK_INT}"
}

actual fun getPlatform(): Platform = AndroidPlatform()

actual fun httpClient(config: HttpClientConfig<*>.() -> Unit): HttpClient = HttpClient(config)
