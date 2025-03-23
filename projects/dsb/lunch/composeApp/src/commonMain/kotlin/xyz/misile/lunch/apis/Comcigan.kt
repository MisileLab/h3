@file:UseSerializers(
    EitherSerializer::class
)

package xyz.misile.lunch.apis

import arrow.core.Either
import arrow.core.serialization.EitherSerializer
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.parameter
import kotlinx.serialization.Serializable
import kotlinx.serialization.UseSerializers
import xyz.misile.lunch.client
import xyz.misile.lunch.getData
import xyz.misile.lunch.putData

@Serializable
data class School(
    val schoolName: String,
    val schoolCode: Either<UInt, String>,
    val region: String,
    val regionCode: UInt?
)

@Serializable
data class ClassList(
    val grade: UInt,
    val classes: Array<UInt>
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as ClassList

        if (grade != other.grade) return false
        if (!classes.contentEquals(other.classes)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = grade.hashCode()
        result = 31 * result + classes.contentHashCode()
        return result
    }
}

@Serializable
data class Timetable(
    val subject: String,
    val teacher: String,
    val changed: Boolean,
    val originalSubject: String?,
    val originalTracker: String?
)

suspend fun comciganSearch(schoolName: String): Array<School> {
    return client.get("/comcigan/search") {
        parameter("schoolName", schoolName)
    }.body()
}

suspend fun getClassList(schoolCode: UInt): Array<ClassList> {
    return client.get("/comcigan/classList") {
        parameter("schoolCode", schoolCode)
    }.body()
}

suspend fun getTimetable(context: Any?, schoolCode: UInt, grade: UInt, classNum: UInt): Array<Array<Timetable>> {
    val cacheKey = "timetable_${schoolCode}_${grade}_${classNum}"
    val cached = getData<Array<Array<Timetable>>>(context, cacheKey)
    if (cached != null) {
        return cached
    }

    val resp: Array<Array<Timetable>> = client.get("/neis/search") {
        parameter("schoolCode", schoolCode)
        parameter("grade", grade)
        parameter("class", classNum)
    }.body()
    putData(context, cacheKey, resp)
    return resp
}
