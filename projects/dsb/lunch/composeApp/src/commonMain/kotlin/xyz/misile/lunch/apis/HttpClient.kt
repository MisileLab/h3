package xyz.misile.lunch.apis

import io.ktor.client.HttpClient
import io.ktor.client.HttpClientConfig
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.serialization.kotlinx.json.json

expect fun httpClient(config: HttpClientConfig<*>.() -> Unit = {}): HttpClient

val client = httpClient {
    install(Logging)
    install(ContentNegotiation) { json() }
    defaultRequest {
        url("https://slunch-v2.ny64.kr")
    }
    expectSuccess = true
}
