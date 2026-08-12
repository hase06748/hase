
import 'package:flutter/services.dart';

/// One ONNX graph as the native side sees it on disk.
class ModelSlot {
  /// Canonical file name, e.g. `hat_x4.onnx`.
  final String name;

  /// Arabic label for the UI.
  final String label;

  /// What the pipeline loses without it.
  final String role;

  /// False when the pipeline degrades gracefully instead of refusing to run.
  final bool required;

  /// Byte size of the reference export.
  final int expected;

  /// Byte size actually on disk, 0 when absent.
  final int size;

  final bool present;

  /// A file is there but its size is outside the accepted window — worse than
  /// missing, because it would fail deep inside ONNX Runtime.
  final bool corrupt;

  const ModelSlot({
    required this.name,
    required this.label,
    required this.role,
    required this.required,
    required this.expected,
    required this.size,
    required this.present,
    required this.corrupt,
  });

  factory ModelSlot.fromMap(Map<dynamic, dynamic> m) => ModelSlot(
        name: (m['name'] as String?) ?? '',
        label: (m['label'] as String?) ?? '',
        role: (m['role'] as String?) ?? '',
        required: (m['required'] as bool?) ?? false,
        expected: (m['expected'] as num?)?.toInt() ?? 0,
        size: (m['size'] as num?)?.toInt() ?? 0,
        present: (m['present'] as bool?) ?? false,
        corrupt: (m['corrupt'] as bool?) ?? false,
      );
}

/// Snapshot of the whole model directory.
class ModelStatus {
  final List<ModelSlot> models;

  /// True once the super-resolver is staged, which is all the engine needs
  /// to start; the other four only enable optional stages.
  final bool coreReady;

  final bool allReady;

  /// Bytes currently held on disk by staged models.
  final int occupied;

  const ModelStatus({
    this.models = const [],
    this.coreReady = false,
    this.allReady = false,
    this.occupied = 0,
  });

  int get presentCount => models.where((m) => m.present).length;

  factory ModelStatus.fromMap(Map<dynamic, dynamic> m) => ModelStatus(
        models: ((m['models'] as List?) ?? const [])
            .map((e) => ModelSlot.fromMap(e as Map<dynamic, dynamic>))
            .toList(),
        coreReady: (m['coreReady'] as bool?) ?? false,
        allReady: (m['allReady'] as bool?) ?? false,
        occupied: (m['occupied'] as num?)?.toInt() ?? 0,
      );
}

/// Outcome of staging a batch of picked files.
class ImportOutcome {
  /// Files that were recognised and staged, as (canonical name, label).
  final List<MapEntry<String, String>> imported;

  /// Files that could not be staged, as (source, reason).
  final List<MapEntry<String, String>> failed;

  final bool coreReady;
  final bool allReady;

  const ImportOutcome({
    this.imported = const [],
    this.failed = const [],
    this.coreReady = false,
    this.allReady = false,
  });

  factory ImportOutcome.fromMap(Map<dynamic, dynamic> m) => ImportOutcome(
        imported: ((m['imported'] as List?) ?? const [])
            .map((e) => MapEntry(
                  ((e as Map)['name'] as String?) ?? '',
                  (e['label'] as String?) ?? '',
                ))
            .toList(),
        failed: ((m['failed'] as List?) ?? const [])
            .map((e) => MapEntry(
                  ((e as Map)['uri'] as String?) ?? '',
                  (e['error'] as String?) ?? 'فشل',
                ))
            .toList(),
        coreReady: (m['coreReady'] as bool?) ?? false,
        allReady: (m['allReady'] as bool?) ?? false,
      );
}

/// Bytes copied so far while a model file is being staged.
class ImportProgress {
  final int index;
  final int count;
  final int copied;

  /// -1 when the provider does not report a size.
  final int total;

  const ImportProgress({
    this.index = 0,
    this.count = 1,
    this.copied = 0,
    this.total = -1,
  });

  double get fraction => total <= 0 ? 0 : (copied / total).clamp(0.0, 1.0);
}

/// Result of decoding a source image on the native side.
class SourceImage {
  final int width;
  final int height;
  final String name;
  final String mime;
  final Uint8List preview;
  final int tiles;

  const SourceImage({
    required this.width,
    required this.height,
    required this.name,
    required this.mime,
    required this.preview,
    required this.tiles,
  });

  int get pixels => width * height;

  factory SourceImage.fromMap(Map<dynamic, dynamic> m) => SourceImage(
        width: m['width'] as int,
        height: m['height'] as int,
        name: (m['name'] as String?) ?? 'image',
        mime: (m['mime'] as String?) ?? 'image/*',
        preview: m['preview'] as Uint8List,
        tiles: (m['tiles'] as int?) ?? 0,
      );
}

/// Result of a completed enhancement run, including what the six-stage
/// pipeline actually found and did.
class EnhanceResult {
  final int width;
  final int height;
  final int srcWidth;
  final int srcHeight;
  final int elapsedMs;
  final Uint8List preview;

  /// One-line Arabic summary of the whole run.
  final String summary;

  /// Stage 1 findings.
  final double noiseSigma;
  final double jpegScore;
  final double blurScore;
  final String blurKind;
  final int faces;

  /// Stage 2 and 4 outcomes.
  final bool cleanupApplied;
  final int facesRestored;

  /// Stage 6 outcomes: how much had to be pulled back, and the worst
  /// face-identity similarity measured (-1 when no face was checked).
  final int cellsReverted;
  final int facesReverted;
  final double identity;

  const EnhanceResult({
    required this.width,
    required this.height,
    required this.srcWidth,
    required this.srcHeight,
    required this.elapsedMs,
    required this.preview,
    this.summary = '',
    this.noiseSigma = 0,
    this.jpegScore = 0,
    this.blurScore = 0,
    this.blurKind = 'NONE',
    this.faces = 0,
    this.cleanupApplied = false,
    this.facesRestored = 0,
    this.cellsReverted = 0,
    this.facesReverted = 0,
    this.identity = -1,
  });

  factory EnhanceResult.fromMap(Map<dynamic, dynamic> m) => EnhanceResult(
        width: m['width'] as int,
        height: m['height'] as int,
        srcWidth: (m['srcWidth'] as int?) ?? 0,
        srcHeight: (m['srcHeight'] as int?) ?? 0,
        elapsedMs: (m['ms'] as int?) ?? 0,
        preview: m['preview'] as Uint8List,
        summary: (m['summary'] as String?) ?? '',
        noiseSigma: (m['noiseSigma'] as num?)?.toDouble() ?? 0,
        jpegScore: (m['jpegScore'] as num?)?.toDouble() ?? 0,
        blurScore: (m['blurScore'] as num?)?.toDouble() ?? 0,
        blurKind: (m['blurKind'] as String?) ?? 'NONE',
        faces: (m['faces'] as int?) ?? 0,
        cleanupApplied: (m['cleanupApplied'] as bool?) ?? false,
        facesRestored: (m['facesRestored'] as int?) ?? 0,
        cellsReverted: (m['cellsReverted'] as int?) ?? 0,
        facesReverted: (m['facesReverted'] as int?) ?? 0,
        identity: (m['identity'] as num?)?.toDouble() ?? -1,
      );
}

class TileProgress {
  final int done;
  final int total;

  /// Which pipeline stage is running, e.g. ANALYZE / CLEANUP / UPSCALE.
  final String phase;

  /// Localised name of that stage, ready to display.
  final String phaseLabel;

  /// Free-form detail from the stage, such as the analysis findings.
  final String note;

  /// Wall time of the most recent tile, used for a live ETA.
  final int tileMs;

  /// Localised thermal state reported by the governor.
  final String thermal;

  /// Predicted thermal headroom in percent, -1 when unavailable.
  final int headroom;

  /// True while the governor is deliberately pacing the engine.
  final bool throttling;

  /// Total time spent cooling down so far.
  final int pausedMs;

  /// Physical device temperature in Celsius.
  final double tempC;

  const TileProgress(
    this.done,
    this.total, {
    this.phase = 'UPSCALE',
    this.phaseLabel = '',
    this.note = '',
    this.tileMs = 0,
    this.thermal = '',
    this.tempC = 0.0,
    this.headroom = -1,
    this.throttling = false,
    this.pausedMs = 0,
  });

  /// True for the stages that dominate the wall clock, so the UI can show an
  /// ETA only where one is meaningful.
  bool get isTiled => phase == 'UPSCALE' || phase == 'CLEANUP';

  double get fraction => total <= 0 ? 0 : done / total;

  /// Remaining time estimated from the latest tile cost.
  Duration get eta {
    if (total <= 0 || done <= 0 || tileMs <= 0) return Duration.zero;
    return Duration(milliseconds: (total - done) * tileMs);
  }
}

/// Everything known about a run before it starts.
class ProcessingPlan {
  final int srcWidth;
  final int srcHeight;
  final int outWidth;
  final int outHeight;
  final int cols;
  final int rows;
  final int tiles;
  final int estimatedMs;
  final int workingMemMb;
  final int bitmapMemMb;

  /// False when the x4 output would hold *less* information than the original,
  /// because the downscale to the working ceiling threw away more than the
  /// upscale can put back.
  final bool realGain;

  /// Fraction of the source's pixels that survive into the run, 0..1.
  final double keptFraction;

  /// True when the working ceiling was raised above the chosen preset because
  /// this device has the memory for it.
  final bool budgetRaised;

  final int totalRamMb;
  final int heapMb;

  const ProcessingPlan({
    required this.srcWidth,
    required this.srcHeight,
    required this.outWidth,
    required this.outHeight,
    required this.cols,
    required this.rows,
    required this.tiles,
    required this.estimatedMs,
    required this.workingMemMb,
    required this.bitmapMemMb,
    required this.realGain,
    required this.keptFraction,
    required this.budgetRaised,
    required this.totalRamMb,
    required this.heapMb,
  });

  factory ProcessingPlan.fromMap(Map<dynamic, dynamic> m) => ProcessingPlan(
        srcWidth: (m['srcWidth'] as int?) ?? 0,
        srcHeight: (m['srcHeight'] as int?) ?? 0,
        outWidth: (m['outWidth'] as int?) ?? 0,
        outHeight: (m['outHeight'] as int?) ?? 0,
        cols: (m['cols'] as int?) ?? 0,
        rows: (m['rows'] as int?) ?? 0,
        tiles: (m['tiles'] as int?) ?? 0,
        estimatedMs: (m['estimatedMs'] as num?)?.toInt() ?? 0,
        workingMemMb: (m['workingMemMb'] as int?) ?? 0,
        bitmapMemMb: (m['bitmapMemMb'] as int?) ?? 0,
        realGain: (m['realGain'] as bool?) ?? true,
        keptFraction: (m['keptFraction'] as num?)?.toDouble() ?? 1.0,
        budgetRaised: (m['budgetRaised'] as bool?) ?? false,
        totalRamMb: (m['totalRamMb'] as int?) ?? 0,
        heapMb: (m['heapMb'] as int?) ?? 0,
      );
}

class DeviceInfo {
  final int cores;
  final int maxMemMb;
  final String backend;

  /// Every model's resolved accelerator, e.g. "SR:NNAPI · تنظيف:XNNPACK".
  final String backends;
  final int tile;
  final int overlap;

  /// Measured cost of one tile on this device, 0 when not benchmarked yet.
  final double msPerTile;

  const DeviceInfo({
    required this.cores,
    required this.maxMemMb,
    required this.backend,
    required this.backends,
    required this.tile,
    required this.overlap,
    required this.msPerTile,
  });

  factory DeviceInfo.fromMap(Map<dynamic, dynamic> m) => DeviceInfo(
        cores: (m['cores'] as int?) ?? 4,
        maxMemMb: (m['maxMemMb'] as int?) ?? 256,
        backend: (m['backend'] as String?) ?? '-',
        backends: (m['backends'] as String?) ?? '-',
        tile: (m['tile'] as int?) ?? 64,
        overlap: (m['overlap'] as int?) ?? 8,
        msPerTile: (m['msPerTile'] as num?)?.toDouble() ?? 0,
      );
}

/// Thin wrapper around the native super-resolution engine.
class SrBridge {
  static const MethodChannel _method =
      MethodChannel('com.photoenhancer.editor/sr');
  static const EventChannel _events =
      EventChannel('com.photoenhancer.editor/progress');

  static Stream<Map<dynamic, dynamic>>? _raw;

  /// The single native event stream, shared by both consumers below. It is
  /// broadcast once and filtered by `kind`, because opening the channel twice
  /// would give the second listener nothing.
  static Stream<Map<dynamic, dynamic>> get _stream {
    _raw ??= _events
        .receiveBroadcastStream()
        .map((e) => e as Map<dynamic, dynamic>)
        .asBroadcastStream();
    return _raw!;
  }

  /// Emits copy progress while [importModels] stages a file.
  static Stream<ImportProgress> get importProgress =>
      _stream.where((m) => m['kind'] == 'import').map((m) => ImportProgress(
            index: (m['index'] as num?)?.toInt() ?? 0,
            count: (m['count'] as num?)?.toInt() ?? 1,
            copied: (m['copied'] as num?)?.toInt() ?? 0,
            total: (m['total'] as num?)?.toInt() ?? -1,
          ));

  /// Emits tile progress while [enhance] runs.
  static Stream<TileProgress> get progress {
    return _stream.where((m) => m['kind'] == null).map((m) {
      return TileProgress(
        (m['done'] as int?) ?? 0,
        (m['total'] as int?) ?? 0,
        phase: (m['phase'] as String?) ?? 'UPSCALE',
        phaseLabel: (m['phaseLabel'] as String?) ?? '',
        note: (m['note'] as String?) ?? '',
        tileMs: (m['tileMs'] as num?)?.toInt() ?? 0,
        thermal: (m['thermal'] as String?) ?? '',
        tempC: (m['tempC'] as num?)?.toDouble() ?? 0.0,
        headroom: (m['headroom'] as int?) ?? -1,
        throttling: (m['throttling'] as bool?) ?? false,
        pausedMs: (m['pausedMs'] as num?)?.toInt() ?? 0,
      );
    });
  }

  /// Loads the ONNX model. Returns the active backend name (NNAPI/CPU).
  static Future<String> init({int threads = 4}) async {
    final r = await _method.invokeMethod<String>('init', {'threads': threads});
    return r ?? 'CPU';
  }

  static Future<Map<String, bool>> checkModels() async {
    final r = await _method.invokeMethod<Map<dynamic, dynamic>>('checkModels');
    return r?.map((key, value) => MapEntry(key.toString(), value as bool)) ?? {};
  }

  /// Full picture of the model directory.
  static Future<ModelStatus> modelStatus() async {
    final r = await _method.invokeMethod<Map<dynamic, dynamic>>('modelStatus');
    return ModelStatus.fromMap(r ?? const {});
  }

  /// Stages picked files. Each is identified by its byte size first and its
  /// name second, so a whole folder can be handed over at once.
  ///
  /// [name] forces every file into one slot; leave it null to auto-detect.
  static Future<ImportOutcome> importModels(
    List<String> uris, {
    String? name,
  }) async {
    final r = await _method.invokeMethod<Map<dynamic, dynamic>>('importModels', {
      'uris': uris,
      if (name != null) 'name': name,
    });
    return ImportOutcome.fromMap(r ?? const {});
  }

  static Future<void> cancelImport() => _method.invokeMethod('cancelImport');

  static Future<void> deleteModel(String name) =>
      _method.invokeMethod('deleteModel', {'name': name});

  static Future<bool> importModel(String name, String uri) async {
    final r = await _method
        .invokeMethod<bool>('importModel', {'name': name, 'uri': uri});
    return r ?? false;
  }

  static Future<SourceImage> loadImage(String uri) async {
    final r = await _method
        .invokeMethod<Map<dynamic, dynamic>>('loadImage', {'uri': uri});
    return SourceImage.fromMap(r!);
  }

  /// Runs the six-stage pipeline. Returns null when cancelled.
  ///
  /// [sharpen] drives stage 5 (adaptive fusion), 0..1. The remaining flags
  /// each disable one optional stage; they default to on because the pipeline
  /// already skips work the analyser says is unnecessary.
  static Future<EnhanceResult?> enhance({
    required int maxPixels,
    double sharpen = 0.35,
    bool cleanup = true,
    bool faceRestore = true,
    double faceStrength = 0.8,
    bool qualityGate = true,
    bool protectSkin = true,
    bool protectSky = true,
    int? threads,
  }) async {
    final r = await _method.invokeMethod<Map<dynamic, dynamic>>('enhance', {
      'maxPixels': maxPixels,
      'sharpen': sharpen,
      'cleanup': cleanup,
      'faceRestore': faceRestore,
      'faceStrength': faceStrength,
      'qualityGate': qualityGate,
      'protectSkin': protectSkin,
      'protectSky': protectSky,
      if (threads != null) 'threads': threads,
    });
    if (r == null) return null;
    return EnhanceResult.fromMap(r);
  }

  /// Describes the run that [enhance] would perform for the given budget.
  static Future<ProcessingPlan> plan({required int maxPixels}) async {
    final r = await _method
        .invokeMethod<Map<dynamic, dynamic>>('plan', {'maxPixels': maxPixels});
    return ProcessingPlan.fromMap(r ?? const {});
  }

  static Future<void> cancel() => _method.invokeMethod('cancel');

  /// Saves the last result to the gallery, returns the stored path.
  static Future<String> save({required String format, int quality = 95}) async {
    final r = await _method.invokeMethod<String>(
        'save', {'format': format, 'quality': quality});
    return r ?? '';
  }

  static Future<void> share({required String format, int quality = 95}) =>
      _method.invokeMethod('share', {'format': format, 'quality': quality});

  static Future<DeviceInfo> deviceInfo() async {
    final r = await _method.invokeMethod<Map<dynamic, dynamic>>('deviceInfo');
    return DeviceInfo.fromMap(r ?? const {});
  }
}
