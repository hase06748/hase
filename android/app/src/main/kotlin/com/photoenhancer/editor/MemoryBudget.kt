package com.photoenhancer.editor

import android.app.ActivityManager
import android.content.Context
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Decides how many source pixels a run may keep, from the memory the device
 * actually has rather than a fixed preset number.
 *
 * Two separate problems are solved here.
 *
 * **1. Fake upscaling.** The pipeline downscales to a preset ceiling and then
 * multiplies by 4. On a 50 MP phone photo with the fast preset that means
 * throwing away 97.6% of the pixels and inventing 16x back, producing a
 * 19.2 MP file that holds *0.38x* the information of the original — a smaller
 * picture, dressed up as a bigger one. Measured across the presets:
 *
 * ```
 *  preset   native   working   output   kept    real gain
 *  سريع      12MP    1.20MP   19.2MP   10.0%      1.60x
 *  سريع      50MP    1.20MP   19.2MP    2.4%      0.38x   <-- net loss
 *  متوازن    50MP    3.20MP   51.2MP    6.4%      1.02x   <-- no gain
 *  أقصى      50MP    8.00MP  128.0MP   16.0%      2.56x
 * ```
 *
 * [effectiveMaxPixels] raises the preset toward what memory allows, so the
 * discarded fraction shrinks; [isRealGain] reports whether any gain is left,
 * so the UI can say so instead of quietly shipping a downscale.
 *
 * **2. Using the RAM that is there.** The presets were sized for a cautious
 * lower bound. A device with 12-16 GB can hold a far larger working set, and
 * capping it at 1.2 MP wastes both the memory and the accelerator.
 *
 * On VRAM: there is no separate video memory to budget for here. The APU and
 * GPU on this class of SoC are integrated and address the same physical DRAM
 * as the CPU, so a single budget covers both; there is no distinct pool that
 * could be filled independently.
 */
object MemoryBudget {

    /** Bytes of Java heap the pipeline may occupy at peak. */
    private const val HEAP_SAFETY = 0.55f

    /** Bytes per source pixel across the whole run, measured from the plan. */
    private const val BYTES_PER_SRC_PIXEL = 84L

    data class Budget(
        /** Source pixels this run may keep. */
        val maxPixels: Long,
        /** The preset the user picked, before any raise. */
        val requestedPixels: Long,
        /** Total device RAM in MB. */
        val totalRamMb: Int,
        /** Java heap ceiling for this process in MB. */
        val heapMb: Int,
        /** True when [maxPixels] was raised above the preset. */
        val raised: Boolean
    )

    /**
     * The ceiling this device can really sustain.
     *
     * The heap limit is the binding constraint, not total RAM: Android caps a
     * process well below the physical amount, and `largeHeap` only lifts that
     * cap partway. Both are read, and the smaller governs.
     */
    fun forDevice(ctx: Context, requested: Long): Budget {
        val rt = Runtime.getRuntime()
        val heapBytes = rt.maxMemory()
        val heapMb = (heapBytes / (1024L * 1024L)).toInt()

        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val mi = ActivityManager.MemoryInfo()
        var totalRamMb = 0
        if (am != null) {
            try {
                am.getMemoryInfo(mi)
                totalRamMb = (mi.totalMem / (1024L * 1024L)).toInt()
            } catch (_: Throwable) {
            }
        }

        // What the heap alone can carry.
        val byHeap = (heapBytes * HEAP_SAFETY).toLong() / BYTES_PER_SRC_PIXEL

        // Never exceed a third of physical RAM even if the heap would allow it:
        // the system still needs room, and being killed mid-run costs the user
        // everything.
        val byRam = if (totalRamMb > 0) {
            (totalRamMb.toLong() * 1024L * 1024L / 3L) / BYTES_PER_SRC_PIXEL
        } else {
            Long.MAX_VALUE
        }

        val ceiling = min(byHeap, byRam).coerceAtLeast(400_000L)
        val allowed = if (ceiling > requested) ceiling else requested.coerceAtMost(ceiling)

        return Budget(
            maxPixels = allowed,
            requestedPixels = requested,
            totalRamMb = totalRamMb,
            heapMb = heapMb,
            raised = allowed > requested
        )
    }

    /**
     * Whether a x[scale] run on [workingPixels], taken from a [nativePixels]
     * source, actually returns more information than the original held.
     *
     * The output is only an honest enlargement when the pixels kept, times the
     * scale factor squared, exceed what came in.
     */
    fun isRealGain(nativePixels: Long, workingPixels: Long, scale: Int): Boolean {
        if (nativePixels <= 0L) return true
        return workingPixels * scale * scale > nativePixels
    }

    /**
     * How much of the source survived, 0..1. Below ~0.25 the run is mostly
     * reconstructing what the downscale discarded.
     */
    fun keptFraction(nativePixels: Long, workingPixels: Long): Float {
        if (nativePixels <= 0L) return 1f
        return (workingPixels.toDouble() / nativePixels.toDouble()).toFloat().coerceIn(0f, 1f)
    }

    /**
     * The largest source that still yields a genuine x[scale] enlargement
     * under [maxPixels]; used to explain the limit to the user.
     */
    fun honestSourceLimit(maxPixels: Long, scale: Int): Long = maxPixels * scale * scale

    /** Side length of a square with [px] pixels, for messages. */
    fun sideOf(px: Long): Int = sqrt(px.toDouble()).toInt()
}
