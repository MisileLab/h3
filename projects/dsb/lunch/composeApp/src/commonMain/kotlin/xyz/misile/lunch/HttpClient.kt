package xyz.misile.lunch

import io.ktor.client.HttpClient
import io.ktor.client.HttpClientConfig
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.serialization.kotlinx.json.json

expect fun httpClient(config: HttpClientConfig<*>.() -> Unit = {}): HttpClient

val client = httpClient {
    install(Logging)
    install(ContentNegotiation) { json() }
    expectSuccess = true
}
