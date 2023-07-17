package xyz.misilelaboratory.unlucky

import org.bukkit.plugin.java.JavaPlugin

const val name = "forget_password"

@Suppress("unused")
class Unlucky: JavaPlugin() {
    override fun onEnable() {
        logger.info("MisileLaboratory | UnluckyPlugin")
    }
}
