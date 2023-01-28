package xyz.misilelaboratory.hardflat

import org.bukkit.entity.LivingEntity
import org.bukkit.event.EventHandler
import org.bukkit.event.Listener
import org.bukkit.event.entity.EntityDamageEvent
import org.bukkit.plugin.java.JavaPlugin

@Suppress("unused")
class HardFlat: JavaPlugin() {
    override fun onEnable() {
        this.logger.info("Enabled plugin")
        this.server.pluginManager.registerEvents(EHandler(), this)
    }
}

class EHandler: Listener {
    @EventHandler
    fun onDamage(e: EntityDamageEvent) {
        if (e is LivingEntity) {
            val ent = e as LivingEntity
            ent.noDamageTicks = 0
        }
    }
}
