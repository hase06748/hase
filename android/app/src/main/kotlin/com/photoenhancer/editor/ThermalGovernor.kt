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
    fun poll() {
        val now = System.currentTimeMillis()
        if (now - lastPoll < 1000L) return
        lastPoll = now

        val pm = power ?: return
        val status = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            runCatching { pm.currentThermalStatus }.getOrDefault(PowerManager.THERMAL_STATUS_NONE)
        } else {
            PowerManager.THERMAL_STATUS_NONE
        }

        // Forecast 20 s ahead: 0 = cold, 1 = about to throttle.
        headroom = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            runCatching { pm.getThermalHeadroom(20) }
                .getOrDefault(Float.NaN)
                .let { if (it.isNaN() || it.isInfinite()) -1f else it }
        } else {
            -1f
        }

        // Status is authoritative, headroom only ever makes us more cautious.
        var f = when {
            status >= PowerManager.THERMAL_STATUS_SEVERE -> 3.0
            status >= PowerManager.THERMAL_STATUS_MODERATE -> 1.2
            status >= PowerManager.THERMAL_STATUS_LIGHT -> 0.4
            else -> 0.0
        }
        var lv = when {
            status >= PowerManager.THERMAL_STATUS_SEVERE -> Level.CRITICAL
            status >= PowerManager.THERMAL_STATUS_MODERATE -> Level.HOT
            status >= PowerManager.THERMAL_STATUS_LIGHT -> Level.WARM
            else -> Level.NORMAL
        }

        if (headroom >= 0f) {
            val hf = when {
                headroom >= 0.98f -> 2.4
                headroom >= 0.90f -> 1.0
                headroom >= 0.80f -> 0.35
                else -> 0.0
            }
            if (hf > f) {
                f = hf
                lv = when {
                    headroom >= 0.98f -> Level.CRITICAL
                    headroom >= 0.90f -> Level.HOT
                    else -> Level.WARM
                }
            }
        }

        // Move gradually so the throughput does not oscillate.
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
