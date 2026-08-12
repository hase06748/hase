import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb, defaultTargetPlatform, TargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'compare_view.dart';
import 'models_page.dart';
import 'sr_bridge.dart';
import 'theme.dart';

enum Stage { idle, loading, ready, working, done }

/// Presets bound the input resolution so a run finishes in predictable time.
class QualityPreset {
  final String id;
  final String label;
  final String hint;
  final int maxPixels;

  const QualityPreset(this.id, this.label, this.hint, this.maxPixels);

  static const fast = QualityPreset('fast', 'سريع', 'حتى 0.3 ميجابكسل', 300000);
  static const balanced =
      QualityPreset('balanced', 'متوازن', 'حتى 0.8 ميجابكسل', 800000);
  static const max = QualityPreset('max', 'أقصى جودة', 'حتى 2 ميجابكسل', 2000000);

  static const all = [fast, balanced, max];

  static QualityPreset byId(String id) =>
      all.firstWhere((p) => p.id == id, orElse: () => balanced);
}

/// Unsharp-mask strength applied to the enlarged image.
class SharpLevel {
  final String id;
  final String label;
  final double amount;

  const SharpLevel(this.id, this.label, this.amount);

  static const soft = SharpLevel('soft', 'ناعمة', 0.0);
  static const normal = SharpLevel('normal', 'متوسطة', 0.35);
  static const high = SharpLevel('high', 'عالية', 0.7);
  static const extra = SharpLevel('extra', 'قصوى', 1.0);

  static const all = [soft, normal, high, extra];

  static SharpLevel byId(String id) =>
      all.firstWhere((s) => s.id == id, orElse: () => high);
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Stage _stage = Stage.idle;
  String _backend = '...';
  bool _engineReady = false;

  /// True when the weights are simply not on disk yet — a setup state, not
  /// an error. Drives the restore call-to-action instead of a red message.
  bool _modelsMissing = false;
  String? _error;

  SourceImage? _source;
  EnhanceResult? _result;

  QualityPreset _preset = QualityPreset.balanced;
  SharpLevel _sharp = SharpLevel.high;
  ProcessingPlan? _plan;
  String _format = 'png';

  // Optional pipeline stages. They default to on: the analyser already skips
  // whatever the image does not need, so switching one off is a deliberate
  // override rather than a performance knob.
  bool _cleanup = true;
  bool _faceRestore = true;
  bool _qualityGate = true;

  /// Every model's resolved accelerator. Only four of the five load lazily
  /// during a run, so this is refreshed once the run finishes.
  String _backends = '';

  TileProgress _progress = const TileProgress(0, 0);
  StreamSubscription<TileProgress>? _sub;
  DateTime? _startedAt;
  Timer? _ticker;
  Duration _elapsed = Duration.zero;

  @override
  void initState() {
    super.initState();
    _sub = SrBridge.progress.listen((p) {
      if (mounted) setState(() => _progress = p);
    });
    WidgetsBinding.instance.addPostFrameCallback((_) => _boot());
  }

  @override
  void dispose() {
    _sub?.cancel();
    _ticker?.cancel();
    super.dispose();
  }

  Future<void> _boot() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedPreset = prefs.getString('preset');
      final savedFormat = prefs.getString('format');
      final savedSharp = prefs.getString('sharp');
      if (mounted) {
        setState(() {
          if (savedPreset != null) _preset = QualityPreset.byId(savedPreset);
          if (savedFormat != null) _format = savedFormat;
          if (savedSharp != null) _sharp = SharpLevel.byId(savedSharp);
          _cleanup = prefs.getBool('cleanup') ?? true;
          _faceRestore = prefs.getBool('faceRestore') ?? true;
          _qualityGate = prefs.getBool('qualityGate') ?? true;
        });
      }
      final info = await SrBridge.deviceInfo();
      // The model is compute bound, so every core is worth using.
      final threads = info.cores > 1 ? info.cores : 1;
      final backend = await SrBridge.init(threads: threads);
      if (!mounted) return;
      setState(() {
        _backend = backend;
        _engineReady = true;
        _modelsMissing = false;
        _error = null;
      });
    } on PlatformException catch (e) {
      if (!mounted) return;
      setState(() {
        _stage = Stage.idle; // Reset stage on error
        if (e.code == 'MODELS_MISSING') {
          _modelsMissing = true;
          _engineReady = false;
          _backend = 'بانتظار النماذج';
          _error = null;
        } else {
          _error = 'تعذّر تحميل النموذج: ${e.message}';
          _backend = 'خطأ';
          _engineReady = false;
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _stage = Stage.idle; // Reset stage on error
        _error = _isNative
            ? 'تعذّر تحميل النموذج: $e'
            : 'هذه معاينة للواجهة فقط.\nمحرّك المعالجة يعمل على أندرويد — ثبّت ملف APK لتجربة التحسين الفعلي.';
        _backend = _isNative ? 'خطأ' : 'معاينة';
        _engineReady = false;
      });
    }

  /// Opens the importer and retries initialisation when something landed, so
  /// a model imported now is usable without restarting the app.
  Future<void> _openModels() async {
    await Navigator.of(context).push<bool>(
      MaterialPageRoute(builder: (_) => const ModelsPage()),
    );
    if (!mounted) return;
    if (!_engineReady) {
      setState(() => _backend = '...');
      await _boot();
    }
  }

  bool get _isNative =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  /// Recomputes the run plan whenever the source or the budget changes.
  Future<void> _refreshPlan() async {
    if (_source == null) return;
    try {
      final p = await SrBridge.plan(maxPixels: _preset.maxPixels);
      if (mounted) setState(() => _plan = p);
    } catch (_) {
      if (mounted) setState(() => _plan = null);
    }
  }

  Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('preset', _preset.id);
    await prefs.setString('format', _format);
    await prefs.setString('sharp', _sharp.id);
    await prefs.setBool('cleanup', _cleanup);
    await prefs.setBool('faceRestore', _faceRestore);
    await prefs.setBool('qualityGate', _qualityGate);
  }

  Future<void> _pick() async {
    try {
      final res = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const [
          'png', 'jpg', 'jpeg', 'heic', 'heif', 'webp', 'bmp'
        ],
        withData: false,
      );
      final path = res?.files.single.path;
      if (path == null) return;

      setState(() {
        _stage = Stage.loading;
        _error = null;
        _result = null;
      });

      final src = await SrBridge.loadImage(Uri.file(path).toString());
      if (!mounted) return;
      setState(() {
        _source = src;
        _stage = Stage.ready;
      });
      await _refreshPlan();
    } on PlatformException catch (e) {
      if (!mounted) return;
      setState(() {
        _stage = _source == null ? Stage.idle : Stage.ready;
        _error = 'تعذّر قراءة الصورة: ${e.message}';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _stage = _source == null ? Stage.idle : Stage.ready;
        _error = 'تعذّر قراءة الصورة';
      });
    }
  }

  Future<void> _enhance() async {
    if (!_engineReady || _source == null) return;
    setState(() {
      _stage = Stage.working;
      _error = null;
      _progress = const TileProgress(0, 0);
      _startedAt = DateTime.now();
      _elapsed = Duration.zero;
    });
    _ticker = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted || _startedAt == null) return;
      setState(() => _elapsed = DateTime.now().difference(_startedAt!));
    });

    try {
      final r = await SrBridge.enhance(
        maxPixels: _preset.maxPixels,
        sharpen: _sharp.amount,
        cleanup: _cleanup,
        faceRestore: _faceRestore,
        qualityGate: _qualityGate,
      );
      _ticker?.cancel();
      if (!mounted) return;
      if (r == null) {
        setState(() => _stage = Stage.ready);
        _toast('تم الإلغاء');
        return;
      }
      setState(() {
        _result = r;
        _stage = Stage.done;
      });
      // Cleanup, faces, detection and identity each resolve their backend the
      // first time they load, which is during the run just finished. Read them
      // now so the report can show what actually executed.
      try {
        final info = await SrBridge.deviceInfo();
        if (mounted) setState(() => _backends = info.backends);
      } catch (_) {
        // Cosmetic only; a failure here must not disturb a finished run.
      }
    } on PlatformException catch (e) {
      _ticker?.cancel();
      if (!mounted) return;
      setState(() {
        _stage = Stage.ready;
        _error = 'فشلت المعالجة: ${e.message}';
      });
    }
  }

  Future<void> _save() async {
    try {
      final path = await SrBridge.save(
          format: _format, quality: _format == 'png' ? 100 : 95);
      _toast('حُفظت في: $path');
    } on PlatformException catch (e) {
      _toast('تعذّر الحفظ: ${e.message}', error: true);
    }
  }

  Future<void> _share() async {
    try {
      await SrBridge.share(
          format: _format, quality: _format == 'png' ? 100 : 95);
    } on PlatformException catch (e) {
      _toast('تعذّرت المشاركة: ${e.message}', error: true);
    }
  }

  void _toast(String msg, {bool error = false}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg, style: const TextStyle(fontSize: 13.5)),
        backgroundColor: error ? AppTheme.danger : AppTheme.surfaceHigh,
        duration: const Duration(seconds: 4),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('محسّن الصور'),
        actions: [
          IconButton(
            icon: const Icon(Icons.download_for_offline_outlined, size: 22),
            tooltip: 'استعادة النماذج',
            onPressed: _openModels,
          ),
          Padding(
            padding: const EdgeInsets.only(left: 14),
            child: Center(child: _backendChip()),
          ),
        ],
      ),
      body: SafeArea(
        top: false,
        child: Column(
          children: [
            Expanded(child: _canvas()),
            _controls(),
          ],
        ),
      ),
    );
  }

  Widget _backendChip() {
    final ok = _engineReady;
    final color = ok ? AppTheme.accent : AppTheme.textMid;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(ok ? Icons.memory : Icons.hourglass_empty, size: 13, color: color),
          const SizedBox(width: 5),
          Text(
            ok ? _backend : 'تحميل',
            style: TextStyle(
                color: color, fontSize: 11, fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }

  Widget _canvas() {
    final src = _source;
    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 4, 14, 10),
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: AppTheme.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppTheme.outline),
        ),
        clipBehavior: Clip.antiAlias,
        child: src == null ? _emptyState() : _imageArea(src),
      ),
    );
  }

  Widget _emptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 30),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 78,
              height: 78,
              decoration: BoxDecoration(
                color: AppTheme.accent.withValues(alpha: 0.1),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.auto_awesome,
                  size: 34, color: AppTheme.accent),
            ),
            const SizedBox(height: 20),
            const Text(
              'ارفع دقة صورك ×4',
              style: TextStyle(fontSize: 19, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            const Text(
              'المعالجة تتم بالكامل على جهازك\nبدون إنترنت وبدون رفع أي بيانات',
              textAlign: TextAlign.center,
              style: TextStyle(
                  color: AppTheme.textMid, fontSize: 13.5, height: 1.6),
            ),
            const SizedBox(height: 22),
            Wrap(
              spacing: 7,
              runSpacing: 7,
              alignment: WrapAlignment.center,
              children: const [
                _Pill('PNG'),
                _Pill('JPG'),
                _Pill('HEIF'),
                _Pill('WEBP'),
              ],
            ),
            if (_modelsMissing) ...[
              const SizedBox(height: 22),
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: const Color(0xFFE0A030).withValues(alpha: 0.10),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                      color: const Color(0xFFE0A030).withValues(alpha: 0.32)),
                ),
                child: Column(
                  children: [
                    const Text(
                      'النماذج غير مثبّتة',
                      style: TextStyle(
                          fontSize: 13.5,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFFE0A030)),
                    ),
                    const SizedBox(height: 7),
                    const Text(
                      'التطبيق يُشحن بدون ملفات ONNX لتقليل حجمه.\n'
                      'استوردها مرة واحدة وسيهيّئها التطبيق تلقائيًا.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                          fontSize: 12, color: AppTheme.textMid, height: 1.55),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _openModels,
                        icon: const Icon(
                            Icons.drive_folder_upload_rounded, size: 19),
                        label: const Text('استعادة النماذج'),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            if (_error != null) ...[
              const SizedBox(height: 20),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: AppTheme.danger, fontSize: 12.5),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _imageArea(SourceImage src) {
    final r = _result;
    final ar = src.width / src.height;

    return Stack(
      children: [
        Positioned.fill(
          child: Padding(
            padding: const EdgeInsets.all(8),
            child: CompareView(
              before: src.preview,
              after: r?.preview,
              aspectRatio: ar <= 0 ? 1 : ar,
            ),
          ),
        ),
        if (_stage == Stage.working)
          Positioned.fill(child: _workingOverlay()),
        if (_stage == Stage.loading)
          const Positioned.fill(
            child: ColoredBox(
              color: Color(0xCC0C0E12),
              child: Center(child: CircularProgressIndicator(strokeWidth: 2.5)),
            ),
          ),
      ],
    );
  }

  Widget _workingOverlay() {
    final p = _progress;
    final f = p.fraction;
    final pct = (f * 100).clamp(0, 100).toStringAsFixed(0);

    // Headline: the stage that is running. Fall back to a neutral label until
    // the first event arrives so the overlay never flashes an empty line.
    final label = p.phaseLabel.isNotEmpty ? p.phaseLabel : 'جارٍ التحضير';
    // Only the tiled stages have a meaningful unit to count.
    final detail = (p.isTiled && p.total > 0)
        ? '$label — المربّع ${p.done} من ${p.total}'
        : label;

    // An ETA is only honest while a tiled stage is grinding through a known
    // number of units; the other stages are short and unmetered.
    String eta = '';
    if (p.isTiled && p.done > 1 && p.total > 0 && p.tileMs > 0) {
      final left = ((p.total - p.done) * p.tileMs / 1000).round();
      eta = left > 0 ? 'المتبقي ≈ ${_fmt(left)}' : '';
    }

    return ColoredBox(
      color: const Color(0xE60C0E12),
      child: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 34),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 92,
                height: 92,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 92,
                      height: 92,
                      child: CircularProgressIndicator(
                        value: f > 0 ? f : null,
                        strokeWidth: 4,
                        backgroundColor: AppTheme.outline,
                      ),
                    ),
                    Text(
                      '$pct%',
                      style: const TextStyle(
                          fontSize: 20, fontWeight: FontWeight.w700),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              Text(
                detail,
                textAlign: TextAlign.center,
                style: const TextStyle(
                    fontSize: 13.5,
                    color: AppTheme.textHi,
                    fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 10),
              _phaseTrail(),
              if (p.note.isNotEmpty) ...[
                const SizedBox(height: 10),
                Text(
                  p.note,
                  textAlign: TextAlign.center,
                  style:
                      const TextStyle(fontSize: 11.5, color: AppTheme.accent),
                ),
              ],
              const SizedBox(height: 6),
              Text(
                eta.isEmpty ? _fmt(_elapsed.inSeconds) : eta,
                style: const TextStyle(fontSize: 12, color: AppTheme.textMid),
              ),
              if (p.thermal.isNotEmpty) ...[
                const SizedBox(height: 14),
                _thermalBadge(),
              ],
              const SizedBox(height: 24),
              OutlinedButton.icon(
                onPressed: () => SrBridge.cancel(),
                icon: const Icon(Icons.stop_rounded, size: 18),
                label: const Text('إلغاء'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// The six stages, in order, as a row of dots. Stages the analyser decided
  /// to skip are simply passed over, so a dot that never lights up is normal.
  static const _phases = <String, String>{
    'ANALYZE': 'تحليل',
    'CLEANUP': 'تنظيف',
    'UPSCALE': 'تكبير',
    'FACES': 'وجوه',
    'FUSION': 'دمج',
    'GATE': 'فحص',
  };

  Widget _phaseTrail() {
    final keys = _phases.keys.toList();
    final current = keys.indexOf(_progress.phase);
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(keys.length, (i) {
        // Anything before the running stage is finished; DONE lights them all.
        final done = _progress.phase == 'DONE' || (current >= 0 && i < current);
        final active = i == current;
        final color = active
            ? AppTheme.accent
            : done
                ? AppTheme.accent.withValues(alpha: 0.55)
                : AppTheme.outline;
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Column(
            children: [
              AnimatedContainer(
                duration: const Duration(milliseconds: 220),
                width: active ? 9 : 6,
                height: active ? 9 : 6,
                decoration: BoxDecoration(color: color, shape: BoxShape.circle),
              ),
              const SizedBox(height: 4),
              Text(
                _phases[keys[i]]!,
                style: TextStyle(
                  fontSize: 8.5,
                  color: active ? AppTheme.accent : AppTheme.textMid,
                  fontWeight: active ? FontWeight.w700 : FontWeight.w400,
                ),
              ),
            ],
          ),
        );
      }),
    );
  }

  /// Live thermal readout: colour follows severity, text explains the pause.
  Widget _thermalBadge() {
    final p = _progress;
    final hot = p.thermal != 'طبيعية';
    final color = switch (p.thermal) {
      'حرجة' => AppTheme.danger,
      'ساخنة' => const Color(0xFFE8A33D),
      'دافئة' => const Color(0xFFD9C45A),
      _ => AppTheme.accent,
    };
    final parts = <String>['حرارة الجهاز: ${p.thermal}'];
    if (p.headroom >= 0) parts.add('السعة الحرارية ${p.headroom}%');
    if (p.throttling) parts.add('تهدئة تلقائية');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(9),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(hot ? Icons.local_fire_department_rounded : Icons.eco_rounded,
              size: 15, color: color),
          const SizedBox(width: 7),
          Flexible(
            child: Text(
              parts.join(' · '),
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 11.5, color: color),
            ),
          ),
        ],
      ),
    );
  }

  String _fmt(int seconds) {
    if (seconds < 60) return '$seconds ثانية';
    final m = seconds ~/ 60;
    final s = seconds % 60;
    return s == 0 ? '$m دقيقة' : '$m د $s ث';
  }

  Widget _controls() {
    final src = _source;
    final busy = _stage == Stage.working || _stage == Stage.loading;

    return Container(
      decoration: const BoxDecoration(
        color: AppTheme.surface,
        border: Border(top: BorderSide(color: AppTheme.outline)),
      ),
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (src != null) _infoRow(src),
          if (src != null) const SizedBox(height: 12),
          if (src != null && _stage != Stage.done) _presetPicker(busy),
          if (src != null && _stage != Stage.done) const SizedBox(height: 10),
          if (src != null && _stage != Stage.done) _sharpPicker(busy),
          if (src != null && _stage != Stage.done) const SizedBox(height: 10),
          if (src != null && _stage != Stage.done) _stagePicker(busy),
          if (src != null && _stage != Stage.done) const SizedBox(height: 10),
          if (_stage == Stage.done && _result != null) _reportRow(),
          if (_stage == Stage.done && _result != null)
            const SizedBox(height: 12),
          if (src != null && _stage != Stage.done && _plan != null) _planRow(),
          if (src != null && _stage != Stage.done && _plan != null)
            const SizedBox(height: 12),
          if (src != null && _stage != Stage.done && _plan == null)
            const SizedBox(height: 2),
          if (_stage == Stage.done) _formatPicker(),
          if (_stage == Stage.done) const SizedBox(height: 12),
          _actions(busy),
        ],
      ),
    );
  }

  Widget _infoRow(SourceImage src) {
    final r = _result;
    final srcW = r?.srcWidth ?? src.width;
    final srcH = r?.srcHeight ?? src.height;
    return Row(
      children: [
        Expanded(
          child: _stat(
            'المصدر',
            '$srcW × $srcH',
            Icons.photo_outlined,
          ),
        ),
        Container(width: 1, height: 30, color: AppTheme.outline),
        Expanded(
          child: _stat(
            r == null ? 'الناتج المتوقع' : 'الناتج',
            '${srcW * 4} × ${srcH * 4}',
            Icons.aspect_ratio,
            accent: r != null,
          ),
        ),
        if (r != null) ...[
          Container(width: 1, height: 30, color: AppTheme.outline),
          Expanded(
            child: _stat(
              'الزمن',
              _fmt((r.elapsedMs / 1000).round()),
              Icons.timer_outlined,
            ),
          ),
        ],
      ],
    );
  }

  Widget _stat(String label, String value, IconData icon,
      {bool accent = false}) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 12, color: AppTheme.textMid),
            const SizedBox(width: 4),
            Text(label,
                style: const TextStyle(fontSize: 10.5, color: AppTheme.textMid)),
          ],
        ),
        const SizedBox(height: 3),
        Text(
          value,
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: accent ? AppTheme.accent : AppTheme.textHi,
          ),
        ),
      ],
    );
  }

  Widget _presetPicker(bool busy) {
    return Row(
      children: QualityPreset.all.map((p) {
        final sel = p.id == _preset.id;
        return Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 3),
            child: InkWell(
              borderRadius: BorderRadius.circular(12),
              onTap: busy
                  ? null
                  : () {
                      setState(() => _preset = p);
                      _persist();
                      _refreshPlan();
                    },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                padding: const EdgeInsets.symmetric(vertical: 9),
                decoration: BoxDecoration(
                  color: sel
                      ? AppTheme.accent.withValues(alpha: 0.14)
                      : AppTheme.surfaceHigh,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: sel ? AppTheme.accent : AppTheme.outline,
                  ),
                ),
                child: Column(
                  children: [
                    Text(
                      p.label,
                      style: TextStyle(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: sel ? AppTheme.accent : AppTheme.textHi,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      p.hint,
                      style: const TextStyle(
                          fontSize: 9.5, color: AppTheme.textMid),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  /// The engine's own plan for the run: grid, output size, time and memory.
  Widget _planRow() {
    final p = _plan!;
    final secs = (p.estimatedMs / 1000).round();
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: AppTheme.surfaceHigh,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.outline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'خطة المعالجة: ${p.rows}×${p.cols} = ${p.tiles} مربّع '
            '← ${p.outWidth}×${p.outHeight}',
            style: const TextStyle(fontSize: 11.5, color: AppTheme.textMid),
          ),
          const SizedBox(height: 3),
          Text(
            'زمن تقديري ≈ ${_fmt(secs)} · ذاكرة عمل ≈ '
            '${p.workingMemMb + p.bitmapMemMb} ميجابايت',
            style: const TextStyle(fontSize: 11.5, color: AppTheme.textMid),
          ),
          if (p.budgetRaised) ...[
            const SizedBox(height: 3),
            Text(
              'رُفع سقف العمل ليستفيد من ذاكرة الجهاز '
              '(${p.totalRamMb} ميجابايت)',
              style: const TextStyle(
                  fontSize: 11.5, color: AppTheme.accent),
            ),
          ],
          // The honest-gain warning. When the downscale discards more than the
          // x4 can rebuild, the output file is larger but carries less than the
          // original — the user deserves to know before spending the minutes.
          if (!p.realGain) ...[
            const SizedBox(height: 6),
            Row(
              children: [
                const Icon(Icons.warning_amber_rounded,
                    size: 14, color: Color(0xFFE0A030)),
                const SizedBox(width: 5),
                Expanded(
                  child: Text(
                    'تكبير غير حقيقي: سيُحتفظ بـ'
                    '${(p.keptFraction * 100).toStringAsFixed(0)}% من بكسلات '
                    'الأصل فقط، فالناتج يحمل معلومات أقل من الصورة الأصلية. '
                    'اختر نمطًا أعلى أو صوّر بدقة أقل.',
                    style: const TextStyle(
                        fontSize: 11, color: Color(0xFFE0A030), height: 1.35),
                  ),
                ),
              ],
            ),
          ] else if (p.keptFraction < 0.999) ...[
            const SizedBox(height: 3),
            Text(
              'يُحتفظ بـ${(p.keptFraction * 100).toStringAsFixed(0)}% من '
              'بكسلات الأصل',
              style: const TextStyle(fontSize: 11.5, color: AppTheme.textMid),
            ),
          ],
        ],
      ),
    );
  }

  Widget _sharpPicker(bool busy) {
    return Row(
      children: [
        const Text('الحدة',
            style: TextStyle(fontSize: 12.5, color: AppTheme.textMid)),
        const Spacer(),
        ...SharpLevel.all.map((s) {
          final sel = s.id == _sharp.id;
          return Padding(
            padding: const EdgeInsets.only(right: 6),
            child: InkWell(
              borderRadius: BorderRadius.circular(9),
              onTap: busy
                  ? null
                  : () {
                      setState(() => _sharp = s);
                      _persist();
                    },
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
                decoration: BoxDecoration(
                  color: sel
                      ? AppTheme.accent.withValues(alpha: 0.14)
                      : AppTheme.surfaceHigh,
                  borderRadius: BorderRadius.circular(9),
                  border: Border.all(
                      color: sel ? AppTheme.accent : AppTheme.outline),
                ),
                child: Text(
                  s.label,
                  style: TextStyle(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    color: sel ? AppTheme.accent : AppTheme.textMid,
                  ),
                ),
              ),
            ),
          );
        }),
      ],
    );
  }

  /// Toggles for the three optional stages. Each one only ever runs when the
  /// analyser also asks for it, so these switches can subtract work but never
  /// force it: turning them all on is the default, safest configuration.
  Widget _stagePicker(bool busy) {
    Widget chip(String label, IconData icon, bool on, VoidCallback onTap) {
      final color = on ? AppTheme.accent : AppTheme.textMid;
      return Expanded(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 3),
          child: InkWell(
            borderRadius: BorderRadius.circular(10),
            onTap: busy ? null : onTap,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 160),
              padding: const EdgeInsets.symmetric(vertical: 8),
              decoration: BoxDecoration(
                color: on
                    ? AppTheme.accent.withValues(alpha: 0.12)
                    : AppTheme.surfaceHigh,
                borderRadius: BorderRadius.circular(10),
                border:
                    Border.all(color: on ? AppTheme.accent : AppTheme.outline),
              ),
              child: Column(
                children: [
                  Icon(icon, size: 15, color: color),
                  const SizedBox(height: 3),
                  Text(
                    label,
                    style: TextStyle(
                      fontSize: 10.5,
                      color: color,
                      fontWeight: on ? FontWeight.w700 : FontWeight.w400,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    return Row(
      children: [
        chip('تنظيف', Icons.auto_fix_high_outlined, _cleanup, () {
          setState(() => _cleanup = !_cleanup);
          _persist();
        }),
        chip('ترميم الوجوه', Icons.face_retouching_natural, _faceRestore, () {
          setState(() => _faceRestore = !_faceRestore);
          _persist();
        }),
        chip('فحص الجودة', Icons.verified_outlined, _qualityGate, () {
          setState(() => _qualityGate = !_qualityGate);
          _persist();
        }),
      ],
    );
  }

  /// What the pipeline actually found and did, shown after a run so the user
  /// can tell an untouched image apart from one the quality gate pulled back.
  Widget _reportRow() {
    final r = _result!;
    final lines = <String>[];
    if (r.summary.isNotEmpty) lines.add(r.summary);

    final gate = <String>[];
    if (r.cellsReverted > 0) gate.add('أُعيدت ${r.cellsReverted} منطقة');
    if (r.facesReverted > 0) gate.add('أُعيد ${r.facesReverted} وجه');
    if (r.identity >= 0) {
      gate.add('تطابق الهوية ${(r.identity * 100).round()}%');
    }
    if (gate.isNotEmpty) lines.add(gate.join(' · '));
    if (_backends.isNotEmpty && _backends != '-') {
      lines.add('المسرّعات: $_backends');
    }

    if (lines.isEmpty) return const SizedBox.shrink();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: AppTheme.surfaceHigh,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.outline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (var i = 0; i < lines.length; i++) ...[
            if (i > 0) const SizedBox(height: 3),
            Text(
              lines[i],
              style:
                  const TextStyle(fontSize: 11.5, color: AppTheme.textMid),
            ),
          ],
        ],
      ),
    );
  }

  Widget _formatPicker() {
    return Row(
      children: [
        const Text('صيغة الحفظ',
            style: TextStyle(fontSize: 12.5, color: AppTheme.textMid)),
        const Spacer(),
        ...['png', 'jpg'].map((f) {
          final sel = f == _format;
          return Padding(
            padding: const EdgeInsets.only(right: 6),
            child: InkWell(
              borderRadius: BorderRadius.circular(9),
              onTap: () {
                setState(() => _format = f);
                _persist();
              },
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 7),
                decoration: BoxDecoration(
                  color: sel
                      ? AppTheme.accent.withValues(alpha: 0.14)
                      : AppTheme.surfaceHigh,
                  borderRadius: BorderRadius.circular(9),
                  border: Border.all(
                      color: sel ? AppTheme.accent : AppTheme.outline),
                ),
                child: Text(
                  f.toUpperCase(),
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: sel ? AppTheme.accent : AppTheme.textHi,
                  ),
                ),
              ),
            ),
          );
        }),
      ],
    );
  }

  Widget _actions(bool busy) {
    if (_stage == Stage.done) {
      return Column(
        children: [
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: _save,
                  icon: const Icon(Icons.download_rounded, size: 19),
                  label: const Text('حفظ'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _share,
                  icon: const Icon(Icons.share_outlined, size: 18),
                  label: const Text('مشاركة'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: busy ? null : _pick,
            icon: const Icon(Icons.add_photo_alternate_outlined, size: 18),
            label: const Text('صورة أخرى'),
          ),
        ],
      );
    }

    if (_source == null) {
      return SizedBox(
        width: double.infinity,
        child: FilledButton.icon(
          onPressed: busy ? null : _pick,
          icon: const Icon(Icons.add_photo_alternate_outlined, size: 20),
          label: const Text('اختر صورة'),
        ),
      );
    }

    // With no weights on disk the primary action becomes the import, since
    // pressing "enhance" could only ever fail.
    if (_modelsMissing) {
      return Row(
        children: [
          Expanded(
            flex: 2,
            child: FilledButton.icon(
              onPressed: _openModels,
              icon: const Icon(Icons.drive_folder_upload_rounded, size: 19),
              label: const Text('استعادة النماذج'),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: OutlinedButton(
              onPressed: busy ? null : _pick,
              child: const Text('تغيير'),
            ),
          ),
        ],
      );
    }

    return Row(
      children: [
        Expanded(
          flex: 2,
          child: FilledButton.icon(
            onPressed: (busy || !_engineReady) ? null : _enhance,
            icon: const Icon(Icons.auto_awesome, size: 19),
            label: Text(_engineReady ? 'تحسين ×4' : 'جارٍ تحميل النموذج'),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: OutlinedButton(
            onPressed: busy ? null : _pick,
            child: const Text('تغيير'),
          ),
        ),
      ],
    );
  }
}

class _Pill extends StatelessWidget {
  final String text;
  const _Pill(this.text);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
      decoration: BoxDecoration(
        color: AppTheme.surfaceHigh,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.outline),
      ),
      child: Text(
        text,
        style: const TextStyle(
            fontSize: 11, color: AppTheme.textMid, fontWeight: FontWeight.w600),
      ),
    );
  }
}
