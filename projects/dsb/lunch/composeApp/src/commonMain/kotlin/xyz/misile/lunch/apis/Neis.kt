@file:UseSerializers(
    EitherSerializer::class
)

package xyz.misile.lunch.apis

import arrow.core.Either
import arrow.core.serialization.EitherSerializer
import io.ktor.client.call.body
import io.ktor.client.plugins.ClientRequestException
import io.ktor.client.request.get
import io.ktor.client.request.parameter
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.Serializable
import kotlinx.serialization.UseSerializers
import xyz.misile.lunch.getData
import xyz.misile.lunch.putData

@Serializable
data class Range(val start: String, val end: String)

@Serializable
data class Schedule(val date: Range, val schedule: String)

@Serializable
data class MealOrigin(val food: String, val origin: String)

@Serializable
data class MealNutrition(val type: String, val amount: String)

@Serializable
data class MealAllergy(val type: String, val code: String)

@Serializable
data class MealItem(val food: String, val allergy: Array<MealAllergy>) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as MealItem

        if (food != other.food) return false
        if (!allergy.contentEquals(other.allergy)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = food.hashCode()
        result = 31 * result + allergy.contentHashCode()
        return result
    }
}

@Serializable
data class Meal(
    val date: String,
    val meal: Array<Either<String, MealItem>>,
    val type: String?,
    val origin: Array<MealOrigin>?,
    val nutrition: Array<MealNutrition>?
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as Meal

        if (date != other.date) return false
        if (!meal.contentEquals(other.meal)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = date.hashCode()
        result = 31 * result + meal.contentHashCode()
        return result
    }
}

suspend fun neisSchoolSearch(schoolName: String): Array<School> {
    return client.get("/neis/search") {
        parameter("schoolName", schoolName)
    }.body()
}

suspend fun getSchedules(
    context: Any?,
    schoolCode: UInt,
    regionCode: String,
    year: String,
    month: String,
    day: String?
): Array<Schedule> {
    val cacheKey = "schedules_${schoolCode}_${regionCode}_${year}_${month}_${day}"
    val cached = getData<Array<Schedule>>(context, cacheKey)
    if (cached != null) {
        return cached
    }

    val resp: Array<Schedule>
    try {
        resp = client.get("/neis/search") {
            parameter("schoolCode", schoolCode)
            parameter("regionCode", regionCode)
            parameter("year", year)
            parameter("month", month)
            parameter("day", day)
        }.body()
    } catch (e: ClientRequestException) {
        if (e.response.status == HttpStatusCode.NotFound) {
            return arrayOf()
        }
        throw e
    }
    putData(context, cacheKey, resp)
    return resp
}

suspend fun getMeal(
    context: Any?,
    schoolCode: UInt,
    regionCode: String,
    year: String,
    month: String,
    day: String?,
    showAllergy: Boolean = false,
    showOrigin: Boolean = false,
    showNutrition: Boolean = false
): Array<Meal> {
    val cacheKey = "meal_${schoolCode}_${regionCode}_${year}_${month}_${day}"
    val cached = getData<Array<Meal>>(context, cacheKey)
    if (cached != null) {
        return cached
    }

    val resp: Array<Meal>
    try {
        resp = client.get("/neis/search") {
            parameter("schoolCode", schoolCode)
            parameter("regionCode", regionCode)
            parameter("year", year)
            parameter("month", month)
            parameter("day", day)
            parameter("showAllergy", showAllergy)
            parameter("showOrigin", showOrigin)
            parameter("showNutrition", showNutrition)
        }.body()
    } catch (e: ClientRequestException) {
        if (e.response.status == HttpStatusCode.NotFound) {
            return arrayOf()
        }
        throw e
    }
    putData(context, cacheKey, resp)
    return resp
}
