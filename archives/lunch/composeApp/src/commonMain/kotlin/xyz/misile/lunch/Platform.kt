package xyz.misile.lunch

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform