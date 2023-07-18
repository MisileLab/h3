package xyz.misilelaboratory.unlucky

import org.bukkit.event.EventHandler
import org.bukkit.event.Listener
import org.bukkit.event.player.PlayerInteractEvent

const val name = "forget_password"

class UnluckyHandler: Listener {
    @EventHandler
    fun onMining(e: PlayerInteractEvent) {
        if (e.player.name == name && (1..100).random() == 1 && e.clickedBlock != null) {
            e.player.isVisualFire = false;
            e.player.fireTicks = 10;
        }
    }
}