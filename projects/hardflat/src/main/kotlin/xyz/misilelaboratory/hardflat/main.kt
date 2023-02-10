package xyz.misilelaboratory.hardflat

import io.papermc.paper.event.player.PrePlayerAttackEntityEvent
import org.bukkit.Material
import org.bukkit.World
import org.bukkit.attribute.Attribute
import org.bukkit.enchantments.Enchantment
import org.bukkit.entity.LivingEntity
import org.bukkit.entity.Player
import org.bukkit.event.EventHandler
import org.bukkit.event.Listener
import org.bukkit.event.block.BlockBreakEvent
import org.bukkit.event.entity.EntityDamageByEntityEvent
import org.bukkit.event.entity.EntityDamageEvent
import org.bukkit.event.entity.EntityShootBowEvent
import org.bukkit.event.entity.PlayerDeathEvent
import org.bukkit.event.inventory.CraftItemEvent
import org.bukkit.event.player.PlayerItemConsumeEvent
import org.bukkit.inventory.ItemStack
import org.bukkit.plugin.java.JavaPlugin
import org.bukkit.potion.PotionEffect
import org.bukkit.potion.PotionEffectType

@Suppress("unused")
class HardFlat: JavaPlugin() {
    override fun onEnable() {
        logger.info("Enabled plugin")
        server.pluginManager.registerEvents(EHandler(), this)
        var a: World? = null
        for (i in server.worlds) {
            if (i.environment == World.Environment.THE_END) {
                a = i
                break
            }
        }
        if (a!!.enderDragonBattle != null) {
            val b = a.enderDragonBattle!!.enderDragon
            if (b != null) {
                val c = b.getAttribute(Attribute.GENERIC_MAX_HEALTH)!!
                c.baseValue = (2000).toDouble()
                b.registerAttribute(c.attribute)
            }
        }
    }
}

class EHandler: Listener {

    private val rawFood = listOf(Material.PORKCHOP, Material.BEEF, Material.CHICKEN, Material.MUTTON, Material.RABBIT,
        Material.COD, Material.SALMON)
    private val attributes = listOf(Attribute.GENERIC_MAX_HEALTH, Attribute.GENERIC_ATTACK_DAMAGE, Attribute.GENERIC_ATTACK_SPEED, Attribute.GENERIC_ARMOR)

    @EventHandler
    fun onDamage(e: EntityDamageEvent) {
        if (e is LivingEntity) {
            val ent = e as LivingEntity
            ent.noDamageTicks = 0
        }
    }

    @EventHandler
    fun onPlayerDamageToEntity(e: EntityDamageByEntityEvent) {
        if (e.entity is Player) {
            val ent = e.entity as Player
            if (ent.isBlocking) {
                if ((1..20).random() == 20) {
                    ent.shieldBlockingDelay = 100
                    ent.clearActiveItem()
                    ent.sendMessage("shield breaked lol")
                }
            }
        } else if (e.damager is Player) {
            if ((1..10).random() == 0) {
                e.isCancelled = true
            }
        }
    }

    @EventHandler
    fun onBreak(e: BlockBreakEvent) {
        val rand = (1..20).random()
        if (rand >= 19) {
            e.isCancelled = true
            e.player.sendMessage("you breaked air lol")
        } else if (rand == 1) {
            e.isDropItems = false
            e.player.sendMessage("no item")
        } else if ((1..10).random() == 1 && (e.block.type == Material.GLOW_BERRIES || e.block.type == Material.SWEET_BERRIES)) {
            e.player.addPotionEffect(PotionEffect(PotionEffectType.WITHER, 2, 1))
        } else if ((1..20).random() == 1 && (e.block.type == Material.DIAMOND_ORE)) {
            val a = attributes.random()
            val b = e.player.getAttribute(a)
            b!!.baseValue = b.baseValue / 100
            e.player.registerAttribute(b.attribute)
            e.player.sendMessage("added ${a.name} 1%")
        }
    }

    @EventHandler
    fun onDeath(e: PlayerDeathEvent) {
        for (i in e.drops) {
            if (!i.enchantments.containsKey(Enchantment.VANISHING_CURSE)) {
                i.addUnsafeEnchantment(Enchantment.VANISHING_CURSE, 1)
            }
        }
    }

    @EventHandler
    fun onAttack(e: PrePlayerAttackEntityEvent) {
        if (e.player.attackCooldown != (0).toFloat()) {
            e.isCancelled = true
            e.player.sendMessage("no attack because has attack cooldown lol")
        }
    }

    @EventHandler
    fun onBowAttack(e: EntityShootBowEvent) {
        if (e.entity is Player) {
            val p = e.entity as Player
            if (p.attackCooldown != (0).toFloat()) {
                e.isCancelled = true
                p.sendMessage("no attack because has attack cooldown lol")
            }
        }
    }

    @EventHandler
    fun onCrafting(e: CraftItemEvent) {
        if ((1..100).random() == 1) {
            e.isCancelled = true
        } else if ((1..100).random() == 1) {
            e.inventory.result = ItemStack(Material.AIR)
            e.inventory.clear()
        }
    }

    @EventHandler
    fun onFoodEat(e: PlayerItemConsumeEvent) {
        if ((1..20).random() == 1) {
            e.isCancelled = true
        } else if (e.item.type in rawFood) {
            e.player.damage((6).toDouble())
        }
    }

    @EventHandler
    fun onFall(e: EntityDamageEvent) {
        var a = e.entity
        if (a is Player && e.cause == EntityDamageEvent.DamageCause.FALL) {
            a = e.entity as Player
            if ((1..20).random() == 1) {
                a.addPotionEffect(PotionEffect(PotionEffectType.SLOW, 3, 1))
            } else if ((1..50).random() == 1) {
                a.addPotionEffect(PotionEffect(PotionEffectType.SLOW, 3, 1))
                a.addPotionEffect(PotionEffect(PotionEffectType.WITHER, 3, 1))
            }
        }
    }
}
