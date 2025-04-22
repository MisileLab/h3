@file:UseSerializers(
    EitherSerializer::class
)

package xyz.misile.lunch.apis

import arrow.core.Either
import arrow.core.serialization.EitherSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.UseSerializers

@Serializable
data class School(
    val schoolName: String,
    val schoolCode: Either<UInt, String>,
    val region: String,
    val regionCode: UInt?
)
