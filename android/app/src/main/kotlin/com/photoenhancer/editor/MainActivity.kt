package com.photoenhancer.editor

import android.content.ClipData
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.PowerManager
import androidx.core.content.FileProvider
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : FlutterActivity() {

    private val channelName = "com.photoenhancer.editor/sr"
    private val eventName = "com.photoenhancer.editor/progress"

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var progressSink: EventChannel.EventSink? = null

    /** Holds the last result so Flutter can save/share without re-sending pixels. */
    private var lastResult: Bitmap? = null
    private var lastSourceName: String = "image"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, eventName).setStreamHandler(
            object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    progressSink = events
                }

                override fun onCancel(arguments: Any?) {
                    progressSink = null
                }
            }
        )

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName).setMethodCallHandler { call, result ->
            when (call.method) {

                "init" -> {
                    val threads = call.argument<Int>("threads") ?: 4
                    scope.launch {
                        // Missing weights are an expected state on a fresh
                        // install, not a failure: the APK ships without the
                        // 396 MB of graphs. Say so precisely so the UI can
                        // send the user to the importer instead of showing a
                        // stack trace.
                        if (!ModelStore.coreReady(applicationContext)) {
                            withContext(Dispatchers.Main) {
                                result.error(
                                    "MODELS_MISSING",
                                    "النماذج غير مثبّتة بعد",
                                    null
                                )
                            }
                            return@launch
                        }
                        try {
                            val backend = SrEngine.initialize(applicationContext, threads)
                            withContext(Dispatchers.Main) { result.success(backend) }
                        } catch (t: Throwable) {
                            withContext(Dispatchers.Main) {
                                result.error("INIT_FAILED", t.message ?: "init failed", null)
                            }
                        }
                    }
                }

                "checkModels" -> {
                    try {
                        result.success(ModelStore.checkModelsStatus(applicationContext))
                    } catch (t: Throwable) {
                        result.error("CHECK_FAILED", t.message, null)
                    }
                }

                /**
                 * Full picture of the model directory: what is present, what
                 * is the wrong size, and whether the engine can start.
                 */
                "modelStatus" -> {
                    try {
                        ModelStore.sweepPartials(applicationContext)
                        result.success(
                            mapOf(
                                "models" to ModelStore.status(applicationContext),
                                "coreReady" to ModelStore.coreReady(applicationContext),
                                "allReady" to ModelStore.allReady(applicationContext),
                                "occupied" to ModelStore.occupiedBytes(applicationContext),
                            )
                        )
                    } catch (t: Throwable) {
                        result.error("CHECK_FAILED", t.message, null)
                    }
                }

                /**
                 * Stages one or more picked files. Each is identified by its
                 * byte size first and its name second, so the user can hand
                 * over the whole folder at once without matching files to
                 * slots by hand.
                 *
                 * `name` may be supplied to force a slot when the user is
                 * importing from a specific row of the manager.
                 */
                "importModels" -> {
                    val uris = call.argument<List<String>>("uris")
                    val forced = call.argument<String>("name")
                    if (uris.isNullOrEmpty()) {
                        result.error("BAD_ARGS", "uris required", null); return@setMethodCallHandler
                    }
                    if (!importing.compareAndSet(false, true)) {
                        result.error("BUSY", "استيراد آخر قيد التنفيذ", null); return@setMethodCallHandler
                    }
                    scope.launch {
                        val imported = ArrayList<Map<String, Any>>()
                        val failed = ArrayList<Map<String, Any>>()
                        try {
                            for ((index, u) in uris.withIndex()) {
                                if (importCancelled.get()) break
                                try {
                                    val res = ModelStore.importFrom(
                                        ctx = applicationContext,
                                        uri = Uri.parse(u),
                                        forceAsset = forced
                                    ) { copied, total ->
                                        emitImport(
                                            mapOf(
                                                "kind" to "import",
                                                "index" to index,
                                                "count" to uris.size,
                                                "copied" to copied,
                                                "total" to total,
                                            )
                                        )
                                    }
                                    imported.add(
                                        mapOf(
                                            "name" to res.asset,
                                            "label" to res.label,
                                            "bytes" to res.bytes,
                                        )
                                    )
                                } catch (t: Throwable) {
                                    failed.add(
                                        mapOf(
                                            "uri" to u,
                                            "error" to (t.message ?: "فشل الاستيراد"),
                                        )
                                    )
                                }
                            }
                            // A model that arrives after the engine gave up
                            // must be picked up, so drop the stale session.
                            if (imported.isNotEmpty()) SrEngine.release()
                            withContext(Dispatchers.Main) {
                                result.success(
                                    mapOf(
                                        "imported" to imported,
                                        "failed" to failed,
                                        "coreReady" to ModelStore.coreReady(applicationContext),
                                        "allReady" to ModelStore.allReady(applicationContext),
                                    )
                                )
                            }
                        } catch (t: Throwable) {
                            withContext(Dispatchers.Main) {
                                result.error("IMPORT_FAILED", t.message ?: "import failed", null)
                            }
                        } finally {
                            importCancelled.set(false)
                            importing.set(false)
                        }
                    }
                }

                /** Legacy single-file entry point, kept for older call sites. */
                "importModel" -> {
                    val modelName = call.argument<String>("name")
                    val uriStr = call.argument<String>("uri")
                    if (uriStr == null) {
                        result.error("BAD_ARGS", "uri required", null); return@setMethodCallHandler
                    }
                    scope.launch {
                        try {
                            ModelStore.importFrom(
                                applicationContext, Uri.parse(uriStr), modelName
                            )
                            SrEngine.release()
                            withContext(Dispatchers.Main) { result.success(true) }
                        } catch (t: Throwable) {
                            withContext(Dispatchers.Main) {
                                result.error("IMPORT_FAILED", t.message ?: "import failed", null)
                            }
                        }
                    }
                }

                "cancelImport" -> {
                    importCancelled.set(true)
                    result.success(true)
                }

                "deleteModel" -> {
                    val n = call.argument<String>("name")
                    if (n == null) {
                        result.error("BAD_ARGS", "name required", null); return@setMethodCallHandler
                    }
                    ModelStore.drop(applicationContext, n)
                    SrEngine.release()
                    result.success(true)
                }

                "loadImage" -> {
                    val uriStr = call.argument<String>("uri")
                    if (uriStr == null) {
                        result.error("BAD_ARGS", "uri required", null); return@setMethodCallHandler
                    }
                    if (busy.get()) {
                        result.error("BUSY", "processing in progress", null); return@setMethodCallHandler
                    }
                    scope.launch {
                        try {
                            val loaded = ImageIo.decode(applicationContext, Uri.parse(uriStr))
                            lastSourceName = loaded.displayName
                            // Never recycle while a run may still be reading them;
                            // `busy` guarantees no job is active at this point.
                            recycleIfUnused(lastResult, loaded.bitmap)
                            lastResult = null
                            recycleIfUnused(pendingSource, loaded.bitmap)
                            val preview = ImageIo.encodePreview(loaded.bitmap, 2_000_000L, 90)
                            pendingSource = loaded.bitmap
                            withContext(Dispatchers.Main) {
                                result.success(
                                    mapOf(
                                        "width" to loaded.bitmap.width,
                                        "height" to loaded.bitmap.height,
                                        "name" to loaded.displayName,
                                        "mime" to loaded.mimeType,
                                        "preview" to preview,
                                        "tiles" to SrEngine.tileCount(loaded.bitmap.width, loaded.bitmap.height),
                                    )
                                )
                            }
                        } catch (t: Throwable) {
                            withContext(Dispatchers.Main) {
                                result.error("DECODE_FAILED", t.message ?: "cannot read image", null)
                            }
                        }
                    }
                }

                "enhance" -> {
                    val src = pendingSource
                    if (src == null) {
                        result.error("NO_IMAGE", "load an image first", null); return@setMethodCallHandler
                    }
                    if (!busy.compareAndSet(false, true)) {
                        result.error("BUSY", "processing in progress", null); return@setMethodCallHandler
                    }
                    val requested = (call.argument<Int>("maxPixels") ?: 1_200_000).toLong()
                    // Raise the preset to what this device's heap and RAM can
                    // really sustain. Keeping more source pixels is the only way
                    // the x4 result carries more information than the original,
                    // instead of reconstructing what a downscale discarded.
                    val budget = MemoryBudget.forDevice(applicationContext, requested)
                    val maxPixels = budget.maxPixels
                    val sharpen = (call.argument<Double>("sharpen") ?: 0.35).toFloat()
                    val opts = Pipeline.Options(
                        maxPixels = maxPixels,
                        sharpen = sharpen,
                        cleanup = call.argument<Boolean>("cleanup") ?: true,
                        faceRestore = call.argument<Boolean>("faceRestore") ?: true,
                        faceStrength = (call.argument<Double>("faceStrength") ?: 0.8).toFloat(),
                        qualityGate = call.argument<Boolean>("qualityGate") ?: true,
                        protectSkin = call.argument<Boolean>("protectSkin") ?: true,
                        protectSky = call.argument<Boolean>("protectSky") ?: true
                    )
                    val threads = call.argument<Int>("threads")
                        ?: Runtime.getRuntime().availableProcessors()
                    scope.launch {
                        var scaled: Bitmap? = null
                        try {
                            val input = ImageIo.downscaleIfNeeded(src, maxPixels)
                            if (input !== src) scaled = input
                            val started = System.currentTimeMillis()
                            var lastEmit = 0L
                            val governor = ThermalGovernor(applicationContext)
                            // Keeps the CPU alive while the screen is off, but
                            // the governor still decides how hard it may work.
                            val wake = (getSystemService(POWER_SERVICE) as PowerManager)
                                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "PhotoEnhancer:run")
                            wake.acquire(2 * 60 * 60 * 1000L)
                            val outcome = try {
                                Pipeline.run(
                                    ctx = applicationContext,
                                    source = input,
                                    options = opts,
                                    threads = threads,
                                    // Stage 1 reads grain and JPEG blocking off
                                    // the untouched decode; the resample above
                                    // would have erased both.
                                    native = if (input !== src) src else null,
                                    governor = governor,
                                    cancelled = { SrEngine.isCancelled() }
                                ) { p ->
                                    val now = System.currentTimeMillis()
                                    // A phase change must never be swallowed by
                                    // throttling, or the UI would look stuck.
                                    val boundary = p.done == p.total || p.done == 0
                                    if (boundary || now - lastEmit >= 150L) {
                                        lastEmit = now
                                        val sink = progressSink
                                        if (sink != null) {
                                            runOnUiThread {
                                                sink.success(
                                                    mapOf(
                                                        "phase" to p.phase.name,
                                                        "phaseLabel" to p.phase.label,
                                                        "done" to p.done,
                                                        "total" to p.total,
                                                        "tileMs" to p.tileMs,
                                                        "thermal" to p.thermal,
                                                        "headroom" to p.headroomPercent,
                                                        "throttling" to p.throttling,
                                                        "pausedMs" to p.pausedMs,
                                                        "note" to p.note,
                                                    )
                                                )
                                            }
                                        }
                                    }
                                }
                            } finally {
                                if (wake.isHeld) wake.release()
                            }
                            if (outcome == null) {
                                withContext(Dispatchers.Main) { result.success(null) }
                                return@launch
                            }
                            val out = outcome.bitmap
                            lastResult?.recycle()
                            lastResult = out
                            // A crisp preview matters: the user judges sharpness
                            // from this image, not from the saved file.
                            val preview = ImageIo.encodePreview(out, 8_000_000L, 96)
                            val elapsed = System.currentTimeMillis() - started
                            val a = outcome.analysis
                            withContext(Dispatchers.Main) {
                                result.success(
                                    mapOf(
                                        "width" to out.width,
                                        "height" to out.height,
                                        "srcWidth" to input.width,
                                        "srcHeight" to input.height,
                                        "ms" to elapsed,
                                        "preview" to preview,
                                        "summary" to outcome.summary(),
                                        "noiseSigma" to a.noiseSigma,
                                        "jpegScore" to a.jpegScore,
                                        "blurScore" to a.blurScore,
                                        "blurKind" to a.blurKind.name,
                                        "faces" to a.faces.size,
                                        "cleanupApplied" to outcome.cleanupApplied,
                                        "facesRestored" to outcome.facesRestored,
                                        "cellsReverted" to (outcome.verdict?.cellsReverted ?: 0),
                                        "facesReverted" to (outcome.verdict?.facesReverted ?: 0),
                                        "identity" to (outcome.verdict?.worstSimilarity ?: -1f),
                                    )
                                )
                            }
                        } catch (t: Throwable) {
                            val msg = if (t is OutOfMemoryError)
                                "الصورة كبيرة على الذاكرة المتاحة، جرّب وضع جودة أقل"
                            else t.message ?: "processing failed"
                            withContext(Dispatchers.Main) {
                                result.error("ENHANCE_FAILED", msg, null)
                            }
                        } finally {
                            scaled?.recycle()
                            busy.set(false)
                        }
                    }
                }

                "plan" -> {
                    val src = pendingSource
                    if (src == null) {
                        result.error("NO_IMAGE", "load an image first", null); return@setMethodCallHandler
                    }
                    val requested = (call.argument<Int>("maxPixels") ?: 1_200_000).toLong()
                    val budget = MemoryBudget.forDevice(applicationContext, requested)
                    val maxPixels = budget.maxPixels
                    val dims = ImageIo.fittedSize(src.width, src.height, maxPixels)
                    val p = SrEngine.plan(dims.first, dims.second)
                    val nativePx = src.width.toLong() * src.height.toLong()
                    val workPx = dims.first.toLong() * dims.second.toLong()
                    result.success(
                        mapOf(
                            "srcWidth" to p.srcWidth,
                            "srcHeight" to p.srcHeight,
                            "outWidth" to p.outWidth,
                            "outHeight" to p.outHeight,
                            "cols" to p.cols,
                            "rows" to p.rows,
                            "tiles" to p.tiles,
                            "estimatedMs" to p.estimatedMs,
                            // Whether this run genuinely enlarges the picture,
                            // or is mostly rebuilding what the downscale lost.
                            "realGain" to MemoryBudget.isRealGain(nativePx, workPx, SrEngine.SCALE),
                            "keptFraction" to MemoryBudget.keptFraction(nativePx, workPx),
                            "budgetRaised" to budget.raised,
                            "budgetMaxPixels" to budget.maxPixels.toInt(),
                            "totalRamMb" to budget.totalRamMb,
                            "heapMb" to budget.heapMb,
                            "workingMemMb" to p.workingMemMb,
                            "bitmapMemMb" to p.bitmapMemMb,
                        )
                    )
                }

                "cancel" -> {
                    SrEngine.requestCancel()
                    result.success(true)
                }

                "save" -> {
                    val bmp = lastResult
                    if (bmp == null) {
                        result.error("NO_RESULT", "nothing to save", null); return@setMethodCallHandler
                    }
                    val fmt = call.argument<String>("format") ?: "png"
                    val q = call.argument<Int>("quality") ?: 95
                    scope.launch {
                        try {
                            val path = ImageIo.saveToGallery(applicationContext, bmp, lastSourceName, fmt, q)
                            withContext(Dispatchers.Main) { result.success(path) }
                        } catch (t: Throwable) {
                            withContext(Dispatchers.Main) {
                                result.error("SAVE_FAILED", t.message ?: "cannot save", null)
                            }
                        }
                    }
                }

                "share" -> {
                    val bmp = lastResult
                    if (bmp == null) {
                        result.error("NO_RESULT", "nothing to share", null); return@setMethodCallHandler
                    }
                    val fmt = call.argument<String>("format") ?: "png"
                    val q = call.argument<Int>("quality") ?: 95
                    scope.launch {
                        try {
                            val ext = if (fmt.equals("png", true)) "png" else "jpg"
                            val bytes = ImageIo.encode(bmp, fmt, q)
                            val f: File = ImageIo.writeShareFile(
                                applicationContext, bytes,
                                lastSourceName.substringBeforeLast('.') + "_x4." + ext
                            )
                            val uri = FileProvider.getUriForFile(
                                applicationContext, "$packageName.fileprovider", f
                            )
                            val intent = Intent(Intent.ACTION_SEND).apply {
                                type = if (ext == "png") "image/png" else "image/jpeg"
                                putExtra(Intent.EXTRA_STREAM, uri)
                                // Some launchers only honour the grant when the
                                // URI is also present in the clip data.
                                clipData = ClipData.newRawUri("image", uri)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            withContext(Dispatchers.Main) {
                                startActivity(Intent.createChooser(intent, "Share enhanced image"))
                                result.success(true)
                            }
                        } catch (t: Throwable) {
                            withContext(Dispatchers.Main) {
                                result.error("SHARE_FAILED", t.message ?: "cannot share", null)
                            }
                        }
                    }
                }

                "deviceInfo" -> {
                    val rt = Runtime.getRuntime()
                    result.success(
                        mapOf(
                            "cores" to rt.availableProcessors(),
                            "maxMemMb" to (rt.maxMemory() / (1024 * 1024)).toInt(),
                            "backend" to SrEngine.backendName(),
                            // Every model's resolved accelerator, so the UI can
                            // show what is *actually* running rather than only
                            // the super-resolver's choice.
                            "backends" to Accelerator.summary(),
                            "tile" to SrEngine.TILE,
                            "overlap" to SrEngine.OVERLAP,
                            "msPerTile" to SrEngine.millisPerTile(),
                        )
                    )
                }

                else -> result.notImplemented()
            }
        }
    }

    /**
     * Import progress rides the same channel as the pipeline's, tagged with a
     * `kind` so the Dart side can route it. Reusing the channel keeps the
     * native surface to one stream instead of two competing sinks.
     */
    private fun emitImport(payload: Map<String, Any>) {
        val sink = progressSink ?: return
        runOnUiThread { runCatching { sink.success(payload) } }
    }

    private fun recycleIfUnused(candidate: Bitmap?, keep: Bitmap) {
        if (candidate != null && candidate !== keep && !candidate.isRecycled) {
            candidate.recycle()
        }
    }

    override fun onDestroy() {
        scope.cancel()
        progressSink = null
        lastResult?.recycle()
        lastResult = null
        pendingSource?.recycle()
        pendingSource = null
        SrEngine.release()
        super.onDestroy()
    }

    private val busy = AtomicBoolean(false)

    /** Serialises imports; two copies into the same slot would race. */
    private val importing = AtomicBoolean(false)
    private val importCancelled = AtomicBoolean(false)

    /** Kept per-instance so nothing survives the activity. */
    private var pendingSource: Bitmap? = null
}
