package com.photoenhancer.editor

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import java.util.EnumSet
import kotlin.math.max

/**
 * One place that decides *how* every ONNX session is executed, so all five
 * models follow the same accelerator ladder instead of four of them quietly
 * running on the CPU.
 *
     * The ladder, best first:
     *
     *   1. NNAPI fp32  — NPU/GPU hardware acceleration via vendor accelerator (MediaTek APU / Qualcomm GPU).
     *   2. NNAPI fp16  — NPU/GPU with relaxed precision (when safe).
     *   3. CPU         — Final fallback when hardware acceleration is unavailable.
 *
 * A note on why there is no QNN or TFLite-GPU rung, since both get asked for:
 *
 *   - QNN (Hexagon/HTP) is Qualcomm silicon only. The target device is
 *     MediaTek, so `libQnnHtp.so` can never load there; it was measured to
 *     fail every probe while adding ~70 MB to the APK, and was removed.
 *     NNAPI is the vendor-neutral door to the very same class of hardware.
 *   - TFLite's GPU delegate belongs to TFLite. These are ONNX graphs run by
 *     ONNX Runtime; a TFLite delegate cannot be attached to them. The GPU is
 *     reached, when the driver exposes it, through NNAPI — which is rung 1.
 */
object Accelerator {

    /**
     * Whether a model may use the fp16 rung.
     *
     * HAT is [Precision.STRICT]: its activations were measured escalating to
     * 23168 at `layers.5`, far past the 65504 fp16 ceiling, and the graph
     * produced its first NaN at node 3036. Anything that must stay exact goes
     * in the same bucket.
     */
    enum class Precision { STRICT, RELAXED }

    private enum class Ep(val label: String) {
        NNAPI_FP32("NPU/GPU (NNAPI)"),
        NNAPI_FP16("NPU/GPU (fp16)"),
        CPU("CPU")
    }

    /** What each model ended up running on, for the UI. */
    private val active = java.util.concurrent.ConcurrentHashMap<String, String>()

    fun backendOf(model: String): String = active[model] ?: "-"

    /** Every resolved backend, e.g. "SR:NNAPI · تنظيف:XNNPACK". */
    fun summary(): String {
        if (active.isEmpty()) return "-"
        return active.entries
            .sortedBy { it.key }
            .joinToString(" · ") { "${it.key}:${it.value}" }
    }

    fun forget(model: String) { active.remove(model) }

    /**
     * Records a backend resolved elsewhere. [SrEngine] keeps its own ladder —
     * it races the providers on a real tile and blacklists any that wedged the
     * device mid-probe — but its answer belongs in the same registry so the UI
     * can show all five models together.
     */
    fun register(model: String, backend: String) { active[model] = backend }

    /**
     * Creates a session on the best rung that actually loads *and* passes
     * [verify]. The verifier is what keeps the accelerated rungs honest: a
     * driver may accept a graph and then return NaN, so each candidate is
     * asked to prove itself on real data before it is kept.
     *
     * @param label   short name shown in the UI, e.g. "SR".
     * @param verify  returns true when the session's output is trustworthy.
     *                Sessions that fail are closed and the ladder continues.
     */
    fun create(
        env: OrtEnvironment,
        modelPath: String,
        label: String,
        threads: Int,
        precision: Precision = Precision.STRICT,
        verify: ((OrtSession) -> Boolean)? = null
    ): OrtSession? {
        val ladder = when (precision) {
            Precision.STRICT -> listOf(Ep.NNAPI_FP32, Ep.CPU)
            Precision.RELAXED -> listOf(Ep.NNAPI_FP32, Ep.NNAPI_FP16, Ep.CPU)
        }

        for (ep in ladder) {
            val s = tryCreate(env, modelPath, ep, threads) ?: continue
            val ok = try {
                verify?.invoke(s) ?: true
            } catch (t: Throwable) {
                false
            }
            if (ok) {
                active[label] = ep.label
                return s
            }
            try { s.close() } catch (_: Throwable) {}
        }
        return null
    }

    private fun tryCreate(
        env: OrtEnvironment,
        modelPath: String,
        ep: Ep,
        threads: Int
    ): OrtSession? {
        return try {
            val opts = OrtSession.SessionOptions()
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setMemoryPatternOptimization(true)
            opts.setInterOpNumThreads(1)
            val t = max(1, threads)

            when (ep) {
                Ep.NNAPI_FP32 -> {
                    opts.setIntraOpNumThreads(t)
                    // No flags: the empty set asks NNAPI for full precision on
                    // whatever accelerator the driver exposes. Any relaxation
                    // belongs on the fp16 rung below, where it is measured.
                    opts.addNnapi(EnumSet.noneOf(NNAPIFlags::class.java))
                }
                Ep.NNAPI_FP16 -> {
                    opts.setIntraOpNumThreads(t)
                    opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                }

                Ep.CPU -> opts.setIntraOpNumThreads(t)
            }
            env.createSession(modelPath, opts)
        } catch (t: Throwable) {
            null
        }
    }
}
