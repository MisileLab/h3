package xyz.misile.lunch

import org.jetbrains.compose.resources.Font
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.compose.material.Typography

import lunch.composeapp.generated.resources.Res
import lunch.composeapp.generated.resources.pretendard_black
import lunch.composeapp.generated.resources.pretendard_bold
import lunch.composeapp.generated.resources.pretendard_thin
import lunch.composeapp.generated.resources.pretendard_extra_light
import lunch.composeapp.generated.resources.pretendard_extra_bold
import lunch.composeapp.generated.resources.pretendard_light
import lunch.composeapp.generated.resources.pretendard_medium
import lunch.composeapp.generated.resources.pretendard_regular
import lunch.composeapp.generated.resources.pretendard_semi_bold

val colors = mapOf(
    "white" to Color(0xFFFEFCFF),
    "background" to Color(0xFF181818),
    "highlight" to Color(0xFF7956FC),
    "highlightLight" to Color(0xFFBAA6FF),
    "card" to Color(0xFF252525),
    "primaryText" to Color(0xFFFEFCFF),
    "secondaryText" to Color(0xFFB0B0B0),
    "border" to Color(0xFF333333),
)

@Composable
fun Font(): FontFamily {
    return FontFamily(
        Font(Res.font.pretendard_thin, FontWeight.Thin),
        Font(Res.font.pretendard_extra_light, FontWeight.ExtraLight),
        Font(Res.font.pretendard_light, FontWeight.Light),
        Font(Res.font.pretendard_regular, FontWeight.Normal),
        Font(Res.font.pretendard_medium, FontWeight.Medium),
        Font(Res.font.pretendard_semi_bold, FontWeight.SemiBold),
        Font(Res.font.pretendard_bold, FontWeight.Bold),
        Font(Res.font.pretendard_extra_bold, FontWeight.ExtraBold),
        Font(Res.font.pretendard_black, FontWeight.Black)
    )
}

@Composable
fun Theme(): Typography {
    val font = Font()
    val color = colors["primaryText"]!!
    return Typography(

    )
}
