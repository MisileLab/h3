package xyz.misile.lunch.components

import androidx.compose.foundation.layout.Row
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.vector.ImageVector

@Composable
fun Card(
    title: String,
    subtitle: String,
    arrow: String,
    titleIcon: ImageVector,
    children: @Composable () -> Unit,
    notificationDot: Boolean
) {
    return Row {

    }
}
