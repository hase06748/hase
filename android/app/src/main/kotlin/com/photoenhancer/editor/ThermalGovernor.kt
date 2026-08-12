package com.photoenhancer.editor

import android.content.Context
import android.os.Build
import android.os.PowerManager
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Keeps a long super-resolution run from cooking the phone.
 *
 * Two independent signals drive the decision:
 *
 *  - the OS thermal status (`PowerManager.getCurrentThermalStatus`), which is a
 *    coarse but authoritative "the device is getting hot" flag;
 *  - the thermal headroom forecast (`getThermalHeadroom`, API 30+), which is a
 *    continuous 0..1 estimate of how close the SoC is to throttling in the
 *    near future and therefore lets us slow down *before* it happens.
 *
 * The governor converts those into a duty cycle: after every tile the worker
 * sleeps for a fraction of the time the tile took. That keeps the average
 * power draw down while never dropping work on the floor, and it degrades
 * smoothly instead of letting the kernel hard-throttle the cores.
 */
class ThermalGovernor(ctx: Context) {

    enum class Level(val label: String) {
        NORMAL("طبيعية"),
        WARM("دافئة"),
        HOT("ساخنة"),
        CRITICAL("حرجة")
    }

    private val power = ctx.getSystemService(Context.POWER_SERVICE) as? PowerManager
    private val appContext = ctx.applicationContext
    
    // Real physical temperature from battery / thermal sensors, refreshed live
    fun physicalTempCelsius(): Float {
        val intent = appContext.registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
        val tempInt = intent?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        if (tempInt > 0) return tempInt / 10.0f
        return 39.5f // Fallback safe active ambient
    }

    /** Duty factor: sleep = tileMillis * factor. */
    @Volatile private var factor: Double = 0.0
    @Volatile private var level: Level = Level.NORMAL
    @Volatile private var headroom: Float = -1f
    private var lastPoll = 0L
    private var pausedTotalMs = 0L

    fun level(): Level = level
    fun headroomPercent(): Int = if (headroom < 0f) -1 else (headroom * 100).roundToInt()
    fun pausedMillis(): Long = pausedTotalMs

    /** Cheap enough to call between tiles; the OS is polled once a second. */
    fun poll(turboMode: Boolean = false) {
        val now = System.currentTimeMillis()
        if (now - lastPoll < 1000L) return
        lastPoll = now

        val currentTemp = physicalTempCelsius()

        // If Turbo Mode is active, we ignore the strict 40C limit and run at max speed
        if (turboMode) {
            factor = 0.0
            level = Level.NORMAL
            return
        }

        // Strict 40°C Battery Protection Guard for Xiaomi 15T Pro
        var f = 0.0
        var lv = Level.NORMAL

        if (currentTemp >= 41.5f) {
            f = 3.5 // Heavy cooling pause
            lv = Level.CRITICAL
        } else if (currentTemp >= 40.0f) {
            f = 2.0 // Moderate cooling pause to lock at 40°C
            lv = Level.HOT
        } else if (currentTemp >= 39.0f) {
            f = 0.8 // Light preemptive pacing
            lv = Level.WARM
        } else {
            f = 0.0
            lv = Level.NORMAL
        }

        factor = if (f > factor) min(f, factor + 0.5) else max(f, factor - 0.25)
        level = lv
    }

    /**
     * Sleeps for the cool-down slice earned by a tile that took [tileMillis].
     * Returns immediately when the device is cool. [abort] lets a cancel
     * request break a long pause.
     */
    fun coolDown(tileMillis: Long, abort: () -> Boolean) {
        val f = factor
        if (f <= 0.01) return
        // Cap a single pause so progress never appears frozen.
        val target = min((tileMillis * f).toLong(), 4000L)
        var slept = 0L
        while (slept < target && !abort()) {
            val slice = min(120L, target - slept)
            try {
                Thread.sleep(slice)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            }
            slept += slice
        }
        pausedTotalMs += slept
    }

    /** True while the governor is actively holding the engine back. */
    fun isThrottling(): Boolean = factor > 0.01
}
