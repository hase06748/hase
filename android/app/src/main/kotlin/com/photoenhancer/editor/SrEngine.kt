package com.photoenhancer.editor

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Tiled x4 super-resolution engine backed by Real_HAT_GAN_SRx4_sharper (ONNX).
 *
 * Model contract (validated at export time):
 *   input : float32 [1,3,64,64]  RGB in 0..1
 *   output: float32 [1,3,256,256] RGB in 0..1
 *
 * This is stage 3 of [Pipeline] and it is deliberately the one stage that is
 * never skipped, softened or approximated:
 *
 *  1. The weights stay float32. An fp16 conversion was built and measured, and
 *     rejected: HAT's internal activations reach ~23 000, and once they pass
 *     fp16's 65 504 ceiling the graph produces NaN. Quality is absolute here,
 *     so the 165 MB fp32 model is what ships.
 *  2. It *measures* every available execution provider on the real device and
 *     keeps the fastest one, rejecting any that emits NaN. Nothing is reported
 *     as the backend unless it actually produced a valid timing.
 *  3. It composes the output band by band, so peak Java heap stays flat
 *     (~60 MB) no matter how large the picture is.
 */
object SrEngine {

    const val TILE = 64
    const val OVERLAP = 12
    const val SCALE = 4

    private const val STRIDE = TILE - 2 * OVERLAP      // 40
    private const val OUT_TILE = TILE * SCALE          // 256
    private const val OUT_STRIDE = STRIDE * SCALE      // 192
    private const val MODEL_ASSET_SOTA = "swinir_x4.onnx"
    private const val MODEL_ASSET_FALLBACK = "hat_x4.onnx"
    private const val PREFS = "sr_engine"
    // v2: the provider list changed (QNN dropped), so old choices are void.
    private const val KEY_EP = "ep_choice_v3"
    private const val KEY_PROBING = "ep_probing"
    private const val KEY_BLOCKED = "ep_blocked"

    /**
     * Candidate backends, in probe order.
     *
     * The Qualcomm-only QNN providers were removed: on a MediaTek Dimensity
     * they can never load, so every launch wasted two failed probes and ~70 MB
     * of packaged libraries. NNAPI is the portable route to a vendor
     * accelerator (MediaTek APU, Google TPU, Exynos NPU); NNAPI_FP32 asks for
     * it *without* granting fp16 relaxation, which matters because this model
     * is numerically unsafe in fp16.
     */
    private enum class Ep(val label: String) {
        NNAPI_FP32("مسرّع NPU/GPU (NNAPI)"),
        NNAPI_FP16("مسرّع NPU/GPU (fp16)"),
        CPU("المعالج الأساسي (CPU)")
    }

    @Volatile private var env: OrtEnvironment? = null
    @Volatile private var session: OrtSession? = null
    @Volatile private var backend: String = "-"
    @Volatile private var msPerTile: Double = 0.0
    @Volatile private var cancelRequested: Boolean = false

    /** Precomputed 256x256 blend window (flattened). */
    private val blendWindow: FloatArray by lazy {
        val n = OUT_TILE
        val w = FloatArray(n)
        for (i in 0 until n) {
            val r = 0.5f * (1f - cos(2.0 * PI * (i + 0.5) / (2.0 * n)).toFloat())
            val rr = 0.5f * (1f - cos(2.0 * PI * (n - i - 0.5) / (2.0 * n)).toFloat())
            w[i] = max(min(r, rr), 1e-3f)
        }
        val out = FloatArray(n * n)
        for (y in 0 until n) {
            val wy = w[y]
            val base = y * n
            for (x in 0 until n) out[base + x] = wy * w[x]
        }
        out
    }

    fun isReady(): Boolean = session != null
    fun backendName(): String = backend
    fun millisPerTile(): Double = msPerTile
    fun requestCancel() { cancelRequested = true }

    /** True once cancellation was asked for; every stage polls this. */
    fun isCancelled(): Boolean = cancelRequested

    /** Cleared by the pipeline before a fresh run begins. */
    fun clearCancel() { cancelRequested = false }

    // ---------------------------------------------------------------- init

    @Synchronized
    fun initialize(ctx: Context, threads: Int): String {
        session?.let { return backend }

        val e = OrtEnvironment.getEnvironment()
        env = e

        val modelPath = ensureModelOnDisk(ctx)
        val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

        // A provider that already won a previous benchmark is trusted directly.
        prefs.getString(KEY_EP, null)?.let { cached ->
            val ep = runCatching { Ep.valueOf(cached) }.getOrNull()
            if (ep != null) {
                val s = tryCreate(e, modelPath, ep, threads)
                if (s != null) {
                    session = s
                    backend = ep.label
                    msPerTile = prefs.getFloat("${KEY_EP}_ms", 0f).toDouble()
                    Accelerator.register("SR", backend)
                    return backend
                }
                prefs.edit().remove(KEY_EP).apply()
            }
        }

        // First launch: race the providers and keep the quickest one.
        // A provider that was still being probed when the app last died gets
        // blacklisted, so a backend that wedges the device cannot brick
        // start-up a second time.
        val blocked = HashSet(prefs.getStringSet(KEY_BLOCKED, emptySet()) ?: emptySet())
        prefs.getString(KEY_PROBING, null)?.let { blocked.add(it) }
        if (blocked.isNotEmpty()) {
            prefs.edit().putStringSet(KEY_BLOCKED, blocked).remove(KEY_PROBING).apply()
        }

        var bestSession: OrtSession? = null
        var bestEp: Ep? = null
        var bestMs = Double.MAX_VALUE

        for (ep in Ep.entries) {
            if (ep != Ep.CPU && blocked.contains(ep.name)) continue
            prefs.edit().putString(KEY_PROBING, ep.name).commit()
            val s = tryCreate(e, modelPath, ep, threads)
            val ms = if (s == null) null else benchmark(e, s)
            prefs.edit().remove(KEY_PROBING).commit()
            if (s == null) continue
            if (ms == null) { runCatching { s.close() }; continue }
            if (ms < bestMs) {
                runCatching { bestSession?.close() }
                bestSession = s
                bestEp = ep
                bestMs = ms
            } else {
                runCatching { s.close() }
            }
            // A provider under 400 ms/tile is fast enough; stop probing.
            if (bestMs < 400.0) break
        }

        if (bestSession == null || bestEp == null) {
            throw IllegalStateException("no usable execution provider")
        }

        session = bestSession
        backend = bestEp.label
        msPerTile = bestMs
        Accelerator.register("SR", backend)
        prefs.edit()
            .putString(KEY_EP, bestEp.name)
            .putFloat("${KEY_EP}_ms", bestMs.toFloat())
            .apply()
        return backend
    }

    private fun ensureModelOnDisk(ctx: Context): String {
        return if (ModelStore.isStaged(ctx, MODEL_ASSET_SOTA) || ModelStore.fileOf(ctx, MODEL_ASSET_SOTA).exists()) {
            ModelStore.ensure(ctx, MODEL_ASSET_SOTA)
        } else {
            ModelStore.ensure(ctx, MODEL_ASSET_FALLBACK)
        }
    }

    private fun tryCreate(
        e: OrtEnvironment,
        modelPath: String,
        ep: Ep,
        threads: Int
    ): OrtSession? = try {
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        opts.setMemoryPatternOptimization(true)
        opts.setInterOpNumThreads(1)

        when (ep) {
            Ep.NNAPI_FP32 -> {
                opts.setIntraOpNumThreads(threads)
                // Full precision on the accelerator. Preferred: it is the only
                // accelerated path that cannot degrade the output.
                opts.addNnapi(EnumSet.noneOf(NNAPIFlags::class.java))
            }
            Ep.NNAPI_FP16 -> {
                opts.setIntraOpNumThreads(threads)
                // Tried only if full precision is refused by the driver. The
                // NaN check in benchmark() is what keeps this honest.
                opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }

            Ep.CPU -> opts.setIntraOpNumThreads(threads)
        }
        e.createSession(modelPath, opts)
    } catch (t: Throwable) {
        null
    }

    /** Runs two warm tiles and returns the best time, or null if it fails. */
    private fun benchmark(e: OrtEnvironment, s: OrtSession): Double? = try {
        val buf = FloatBuffer.allocate(3 * TILE * TILE)
        val arr = buf.array()
        for (i in arr.indices) arr[i] = ((i * 37) % 255) / 255f
        val shape = longArrayOf(1, 3, TILE.toLong(), TILE.toLong())
        val inName = s.inputNames.iterator().next()

        var best = Double.MAX_VALUE
        var sane = false
        for (run in 0 until 3) {
            buf.rewind()
            val t0 = System.nanoTime()
            OnnxTensor.createTensor(e, buf, shape).use { tensor ->
                s.run(mapOf(inName to tensor)).use { res ->
                    val fb = (res[0] as OnnxTensor).floatBuffer
                    // Reject any provider that emits NaN or nonsense. A single
                    // sample is not enough: an fp16 driver typically overflows
                    // in part of the tile only, so spread the probes out.
                    val cap = fb.capacity()
                    var ok = cap > 0
                    var i = 0
                    while (i < cap && ok) {
                        val v = fb.get(i)
                        if (v.isNaN() || v.isInfinite() || v < -0.5f || v > 1.5f) ok = false
                        i += 617   // a prime stride, so no channel is skipped
                    }
                    if (ok) sane = true else sane = false
                }
            }
            val ms = (System.nanoTime() - t0) / 1e6
            if (run > 0 && ms < best) best = ms   // ignore the warm-up run
        }
        if (sane && best < Double.MAX_VALUE) best else null
    } catch (t: Throwable) {
        null
    }

    @Synchronized
    fun release() {
        runCatching { session?.close() }
        session = null
        backend = "-"
    }

    // ------------------------------------------------------------- tiling

    /**
     * Everything about a run that can be known before it starts: how many
     * tiles, how long it should take and how much memory it will hold.
     * Computing this up front keeps the pipeline predictable and lets the UI
     * warn the user instead of discovering trouble halfway through.
     */
    data class Plan(
        val srcWidth: Int,
        val srcHeight: Int,
        val outWidth: Int,
        val outHeight: Int,
        val cols: Int,
        val rows: Int,
        val tiles: Int,
        val estimatedMs: Long,
        val workingMemMb: Int,
        val bitmapMemMb: Int
    )

    fun plan(w: Int, h: Int): Plan {
        val xs = originList(w)
        val ys = originList(h)
        val ow = max(w, TILE) * SCALE
        val tiles = xs.size * ys.size
        val perTile = if (msPerTile > 1.0) msPerTile else 2500.0
        // Rolling band + flush buffer + planar source, in bytes.
        val working = 4L * ow * OUT_TILE * 4L + ow.toLong() * OUT_TILE * 4L +
            3L * max(w, TILE) * max(h, TILE) * 4L
        return Plan(
            srcWidth = w,
            srcHeight = h,
            outWidth = w * SCALE,
            outHeight = h * SCALE,
            cols = xs.size,
            rows = ys.size,
            tiles = tiles,
            estimatedMs = (tiles * perTile).toLong(),
            workingMemMb = (working / (1024 * 1024)).toInt(),
            bitmapMemMb = ((w.toLong() * SCALE * h * SCALE * 4L) / (1024 * 1024)).toInt()
        )
    }

    fun tileCount(w: Int, h: Int): Int = originList(w).size * originList(h).size

    private fun originList(size: Int): List<Int> {
        val s = max(size, TILE)
        if (s <= TILE) return listOf(0)
        val list = ArrayList<Int>()
        var p = 0
        while (p + TILE <= s) { list.add(p); p += STRIDE }
        if (list.isEmpty() || list.last() != s - TILE) list.add(s - TILE)
        return list
    }

    /** One report per processed tile, consumed by the UI layer. */
    data class Tick(
        val done: Int,
        val total: Int,
        val tileMs: Long,
        val thermal: String,
        val headroomPercent: Int,
        val throttling: Boolean,
        val pausedMs: Long
    )

    /**
     * Upscales [src] by 4x.
     *
     * The run is executed as an ordered sequence — pad, tile row, blend,
     * flush — with a thermal governor inserted between tiles so the device
     * paces itself instead of being throttled by the kernel. Sharpening is
     * *not* done here: it belongs to stage 5 ([DetailFusion]), which has the
     * edge and skin masks needed to do it without wrecking smooth areas.
     *
     * @return the enlarged bitmap, or null when cancelled.
     */
    fun upscale(
        src: Bitmap,
        governor: ThermalGovernor?,
        onProgress: (Tick) -> Unit
    ): Bitmap? {
        // Note: the flag is cleared by Pipeline.run, not here — stages that
        // run before this one must be cancellable too.
        val sess = session ?: throw IllegalStateException("engine not initialized")
        val e = env ?: throw IllegalStateException("engine not initialized")

        val sw = src.width
        val sh = src.height
        val pw = max(sw, TILE)
        val ph = max(sh, TILE)

        // Planar float source, padded by edge replication.
        val plane = FloatArray(3 * pw * ph)
        run {
            val row = IntArray(sw)
            val area = pw * ph
            for (y in 0 until ph) {
                val sy = if (y < sh) y else sh - 1
                src.getPixels(row, 0, sw, 0, sy, sw, 1)
                val base = y * pw
                for (x in 0 until pw) {
                    val c = row[if (x < sw) x else sw - 1]
                    val i = base + x
                    plane[i] = ((c shr 16) and 0xFF) / 255f
                    plane[area + i] = ((c shr 8) and 0xFF) / 255f
                    plane[2 * area + i] = (c and 0xFF) / 255f
                }
            }
        }
        val area = pw * ph

        val ow = pw * SCALE
        val fw = sw * SCALE
        val fh = sh * SCALE

        val xs = originList(pw)
        val ys = originList(ph)
        val total = xs.size * ys.size
        var done = 0

        // Rolling band: only OUT_TILE output rows are ever live at once.
        val bandRows = OUT_TILE
        val acc = FloatArray(3 * ow * bandRows)
        val wsum = FloatArray(ow * bandRows)
        val bandArea = ow * bandRows

        val result = Bitmap.createBitmap(fw, fh, Bitmap.Config.ARGB_8888)
        // The final band flushes a whole tile height, so size for the maximum.
        val flushBuf = IntArray(ow * OUT_TILE)

        val alphaRef = Bitmap.createScaledBitmap(src, fw, fh, true)
        val alphaRow = IntArray(fw)

        val inBuf = FloatBuffer.allocate(3 * TILE * TILE)
        val shape = longArrayOf(1, 3, TILE.toLong(), TILE.toLong())
        val inName = sess.inputNames.iterator().next()

        for ((rowIdx, ty) in ys.withIndex()) {
            val bandTop = ty * SCALE          // output row of this band's row 0

            for (tx in xs) {
                if (cancelRequested) { result.recycle(); return null }

                val tileStart = System.currentTimeMillis()
                val arr = inBuf.array()
                for (c in 0 until 3) {
                    val cs = c * area
                    val cd = c * TILE * TILE
                    for (y in 0 until TILE) {
                        System.arraycopy(plane, cs + (ty + y) * pw + tx, arr, cd + y * TILE, TILE)
                    }
                }
                inBuf.rewind()

                OnnxTensor.createTensor(e, inBuf, shape).use { tensor ->
                    sess.run(mapOf(inName to tensor)).use { res ->
                        val ob = (res[0] as OnnxTensor).floatBuffer
                        val ox = tx * SCALE
                        for (c in 0 until 3) {
                            val cAcc = c * bandArea
                            val cOff = c * OUT_TILE * OUT_TILE
                            for (y in 0 until OUT_TILE) {
                                val dst = cAcc + y * ow + ox
                                val srcRow = cOff + y * OUT_TILE
                                val wRow = y * OUT_TILE
                                for (x in 0 until OUT_TILE) {
                                    acc[dst + x] += ob.get(srcRow + x) * blendWindow[wRow + x]
                                }
                            }
                        }
                        for (y in 0 until OUT_TILE) {
                            val dst = y * ow + ox
                            val wRow = y * OUT_TILE
                            for (x in 0 until OUT_TILE) wsum[dst + x] += blendWindow[wRow + x]
                        }
                    }
                }
                done++
                val tileMs = System.currentTimeMillis() - tileStart

                // Pace the device: measure, then rest proportionally.
                governor?.poll()
                onProgress(
                    Tick(
                        done = done,
                        total = total,
                        tileMs = tileMs,
                        thermal = (governor?.level() ?: ThermalGovernor.Level.NORMAL).label,
                        headroomPercent = governor?.headroomPercent() ?: -1,
                        throttling = governor?.isThrottling() ?: false,
                        pausedMs = governor?.pausedMillis() ?: 0L
                    )
                )
                governor?.coolDown(tileMs) { cancelRequested }
            }

            // Rows above the next band's top are final: write them out.
            // The last origin can sit closer than STRIDE, so derive the step
            // from the actual origins instead of assuming a fixed stride.
            val last = rowIdx == ys.size - 1
            val settled = if (last) OUT_TILE else (ys[rowIdx + 1] - ty) * SCALE
            flushBand(result, acc, wsum, flushBuf, ow, bandArea, bandTop, settled, fw, fh, alphaRef, alphaRow)

            if (!last) {
                // Slide the window: carry the still-open rows to the front.
                val carryLen = (OUT_TILE - settled) * ow
                val shift = settled * ow
                for (c in 0 until 3) {
                    val base = c * bandArea
                    System.arraycopy(acc, base + shift, acc, base, carryLen)
                    java.util.Arrays.fill(acc, base + carryLen, base + bandArea, 0f)
                }
                System.arraycopy(wsum, shift, wsum, 0, carryLen)
                java.util.Arrays.fill(wsum, carryLen, bandArea, 0f)
            }
        }

        alphaRef.recycle()
        if (cancelRequested) { result.recycle(); return null }
        return result
    }

    /** Normalises [rows] rows of the band and copies them into the bitmap. */
    private fun flushBand(
        dst: Bitmap,
        acc: FloatArray,
        wsum: FloatArray,
        buf: IntArray,
        ow: Int,
        bandArea: Int,
        bandTop: Int,
        rows: Int,
        fw: Int,
        fh: Int,
        alphaRef: Bitmap,
        alphaRow: IntArray
    ) {
        if (bandTop >= fh) return
        val n = min(rows, fh - bandTop)
        if (n <= 0) return
        for (y in 0 until n) {
            val targetY = bandTop + y
            alphaRef.getPixels(alphaRow, 0, fw, 0, targetY, fw, 1)
            val sBase = y * ow
            val dBase = y * fw
            for (x in 0 until fw) {
                val a = (alphaRow[x] ushr 24) and 0xFF
                val i = sBase + x
                val w = if (wsum[i] > 1e-6f) wsum[i] else 1e-6f
                buf[dBase + x] = (a shl 24) or
                    (clamp255(acc[i] / w) shl 16) or
                    (clamp255(acc[bandArea + i] / w) shl 8) or
                    clamp255(acc[2 * bandArea + i] / w)
            }
        }
        dst.setPixels(buf, 0, fw, 0, bandTop, fw, n)
    }

    private fun clamp255(v: Float): Int {
        val x = (v * 255f).roundToInt()
        return if (x < 0) 0 else if (x > 255) 255 else x
    }
}
