import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'sr_bridge.dart';
import 'theme.dart';

/// Restores the five ONNX graphs the pipeline runs on.
///
/// The APK ships without them on purpose: together they are ~396 MB, which
/// would be an absurd download for an app whose code is under a megabyte. So
/// the user hands the files over once, from wherever they keep them, and this
/// screen works out what each one is.
///
/// The whole design assumes the user does *not* know which file feeds which
/// stage. They pick everything at once and the native side identifies each by
/// byte size, falling back to the name. Nothing has to be matched by hand.
class ModelsPage extends StatefulWidget {
  const ModelsPage({super.key});

  @override
  State<ModelsPage> createState() => _ModelsPageState();
}

class _ModelsPageState extends State<ModelsPage> {
  ModelStatus _status = const ModelStatus();
  bool _loading = true;
  bool _importing = false;

  /// Which slot the running import is filling, empty while auto-detecting.
  String _importLabel = '';
  ImportProgress? _progress;
  StreamSubscription<ImportProgress>? _sub;

  @override
  void initState() {
    super.initState();
    _sub = SrBridge.importProgress.listen((p) {
      if (mounted) setState(() => _progress = p);
    });
    _refresh();
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    try {
      final s = await SrBridge.modelStatus();
      if (mounted) setState(() { _status = s; _loading = false; });
    } catch (_) {
      if (mounted) setState(() => _loading = false);
    }
  }

  // ------------------------------------------------------------- importing

  /// Picks any number of files and lets the native side sort them out.
  Future<void> _importAuto() async {
    final res = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      type: FileType.any,
      withData: false,
    );
    final paths = res?.files
            .map((f) => f.path)
            .whereType<String>()
            .map((p) => Uri.file(p).toString())
            .toList() ??
        const <String>[];
    if (paths.isEmpty) return;
    await _run(paths, null, '');
  }

  /// Fills one specific slot, used when auto-detection refused a file.
  Future<void> _importInto(ModelSlot slot) async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.any,
      withData: false,
    );
    final p = res?.files.single.path;
    if (p == null) return;
    await _run([Uri.file(p).toString()], slot.name, slot.label);
  }

  Future<void> _run(List<String> uris, String? force, String label) async {
    setState(() {
      _importing = true;
      _importLabel = label;
      _progress = null;
    });
    try {
      final out = await SrBridge.importModels(uris, name: force);
      await _refresh();
      if (!mounted) return;

      if (out.imported.isNotEmpty) {
        final names = out.imported.map((e) => e.value).join('، ');
        _toast('تم تثبيت: $names');
      }
      for (final f in out.failed) {
        _toast(f.value, error: true);
      }
      if (out.imported.isEmpty && out.failed.isEmpty) {
        _toast('لم يُستورد أي ملف', error: true);
      }
      // The engine can only start once the super-resolver is on disk; tell
      // the caller so it can retry initialisation without a restart.
      if (out.coreReady && mounted) Navigator.of(context).maybePop(true);
    } on PlatformException catch (e) {
      _toast(e.message ?? 'فشل الاستيراد', error: true);
    } finally {
      if (mounted) {
        setState(() {
          _importing = false;
          _progress = null;
        });
      }
    }
  }

  Future<void> _delete(ModelSlot slot) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        backgroundColor: AppTheme.surfaceHigh,
        title: Text('حذف ${slot.label}؟', style: const TextStyle(fontSize: 15)),
        content: Text(
          'سيتحرّر ${_human(slot.size)} من التخزين. يمكنك استيراده مجددًا لاحقًا.',
          style: const TextStyle(fontSize: 13, color: AppTheme.textMid),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c, false),
              child: const Text('إلغاء')),
          TextButton(
            onPressed: () => Navigator.pop(c, true),
            child: const Text('حذف', style: TextStyle(color: AppTheme.danger)),
          ),
        ],
      ),
    );
    if (ok != true) return;
    await SrBridge.deleteModel(slot.name);
    await _refresh();
  }

  void _toast(String msg, {bool error = false}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg, style: const TextStyle(fontSize: 13)),
      backgroundColor: error ? AppTheme.danger : AppTheme.surfaceHigh,
      duration: const Duration(seconds: 4),
    ));
  }

  static String _human(int b) {
    if (b >= 1 << 30) return '${(b / 1073741824).toStringAsFixed(2)} جيجا';
    if (b >= 1 << 20) return '${(b / 1048576).toStringAsFixed(1)} ميجا';
    if (b >= 1 << 10) return '${(b / 1024).toStringAsFixed(0)} كيلو';
    return '$b بايت';
  }

  // ----------------------------------------------------------------- build

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('استعادة النماذج'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded, size: 21),
            tooltip: 'تحديث',
            onPressed: _importing ? null : _refresh,
          ),
        ],
      ),
      body: SafeArea(
        top: false,
        child: _loading
            ? const Center(child: CircularProgressIndicator(strokeWidth: 2.4))
            : Column(
                children: [
                  Expanded(
                    child: ListView(
                      padding: const EdgeInsets.fromLTRB(14, 10, 14, 6),
                      children: [
                        _summary(),
                        const SizedBox(height: 12),
                        ..._status.models.map(_slotCard),
                        const SizedBox(height: 10),
                        _help(),
                      ],
                    ),
                  ),
                  _bottomBar(),
                ],
              ),
      ),
    );
  }

  Widget _summary() {
    final total = _status.models.length;
    final have = _status.presentCount;
    final ready = _status.coreReady;
    final color = ready ? AppTheme.accent : const Color(0xFFE0A030);

    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.09),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withValues(alpha: 0.32)),
      ),
      child: Row(
        children: [
          Icon(
            ready ? Icons.verified_rounded : Icons.downloading_rounded,
            color: color,
            size: 30,
          ),
          const SizedBox(width: 13),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  ready ? 'المحرّك جاهز للعمل' : 'النماذج غير مكتملة',
                  style: TextStyle(
                      fontSize: 14.5, fontWeight: FontWeight.w700, color: color),
                ),
                const SizedBox(height: 4),
                Text(
                  ready
                      ? '$have من $total مثبّت · ${_human(_status.occupied)} على التخزين'
                      : 'مثبّت $have من $total — استورد ملفات ONNX لتفعيل التحسين',
                  style: const TextStyle(
                      fontSize: 11.8, color: AppTheme.textMid, height: 1.4),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _slotCard(ModelSlot m) {
    final Color color;
    final IconData icon;
    final String state;
    if (m.present) {
      color = AppTheme.accent;
      icon = Icons.check_circle_rounded;
      state = 'مثبّت · ${_human(m.size)}';
    } else if (m.corrupt) {
      color = AppTheme.danger;
      icon = Icons.error_rounded;
      state = 'ملف تالف (${_human(m.size)}) — أعد الاستيراد';
    } else {
      color = m.required ? AppTheme.danger : AppTheme.textMid;
      icon = m.required
          ? Icons.priority_high_rounded
          : Icons.radio_button_unchecked_rounded;
      state = 'غير مثبّت · المتوقّع ${_human(m.expected)}';
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 9),
      padding: const EdgeInsets.fromLTRB(13, 12, 8, 12),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: m.present
              ? AppTheme.accent.withValues(alpha: 0.28)
              : AppTheme.outline,
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 1),
            child: Icon(icon, color: color, size: 19),
          ),
          const SizedBox(width: 11),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Flexible(
                      child: Text(
                        m.label,
                        style: const TextStyle(
                            fontSize: 13.5, fontWeight: FontWeight.w700),
                      ),
                    ),
                    if (m.required) ...[
                      const SizedBox(width: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 1.5),
                        decoration: BoxDecoration(
                          color: AppTheme.danger.withValues(alpha: 0.14),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: const Text('إلزامي',
                            style: TextStyle(
                                fontSize: 9.5,
                                color: AppTheme.danger,
                                fontWeight: FontWeight.w700)),
                      ),
                    ],
                  ],
                ),
                const SizedBox(height: 3),
                Text(m.role,
                    style: const TextStyle(
                        fontSize: 11.3, color: AppTheme.textMid, height: 1.35)),
                const SizedBox(height: 5),
                Text(m.name,
                    style: const TextStyle(
                        fontSize: 10.5, color: AppTheme.textMid, height: 1.2)),
                const SizedBox(height: 3),
                Text(state,
                    style: TextStyle(
                        fontSize: 11, color: color, fontWeight: FontWeight.w600)),
              ],
            ),
          ),
          if (m.present)
            IconButton(
              icon: const Icon(Icons.delete_outline_rounded, size: 19),
              color: AppTheme.textMid,
              tooltip: 'حذف',
              onPressed: _importing ? null : () => _delete(m),
            )
          else
            TextButton(
              onPressed: _importing ? null : () => _importInto(m),
              style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  minimumSize: const Size(0, 34)),
              child: const Text('استورد', style: TextStyle(fontSize: 12)),
            ),
        ],
      ),
    );
  }

  Widget _help() {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.outline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          Row(
            children: [
              Icon(Icons.info_outline_rounded,
                  size: 16, color: AppTheme.textMid),
              SizedBox(width: 7),
              Text('كيف تعمل الاستعادة',
                  style: TextStyle(fontSize: 12.5, fontWeight: FontWeight.w700)),
            ],
          ),
          SizedBox(height: 9),
          Text(
            'التطبيق يُشحن بدون النماذج لأن حجمها مجتمعة يقارب 396 ميجابايت.\n\n'
            'اضغط «استيراد الملفات» واختر ملفات ONNX دفعة واحدة — التطبيق '
            'يتعرّف على كل ملف من حجمه بالبايت ثم من اسمه، وينسخه إلى مكانه '
            'الصحيح ويهيّئه للعمل تلقائيًا. لا حاجة لمطابقة الملفات يدويًا.\n\n'
            'النسخ يتم إلى ملف مؤقّت ولا يُعتمد إلا بعد اكتماله، فلا يبقى '
            'نموذج ناقص إذا انقطعت العملية.\n\n'
            'نموذج التكبير وحده يكفي لبدء التشغيل؛ البقية تُفعّل مراحل '
            'اختيارية (تنظيف، ترميم وجوه، فحص هوية).',
            style: TextStyle(
                fontSize: 11.5, color: AppTheme.textMid, height: 1.65),
          ),
        ],
      ),
    );
  }

  Widget _bottomBar() {
    final p = _progress;
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 11, 14, 14),
      decoration: const BoxDecoration(
        color: AppTheme.surface,
        border: Border(top: BorderSide(color: AppTheme.outline)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_importing) ...[
            Row(
              children: [
                Expanded(
                  child: Text(
                    _importLabel.isEmpty
                        ? 'جارٍ نسخ الملفات…'
                        : 'جارٍ نسخ $_importLabel…',
                    style: const TextStyle(fontSize: 12),
                  ),
                ),
                if (p != null && p.count > 1)
                  Text('${p.index + 1}/${p.count}',
                      style: const TextStyle(
                          fontSize: 11.5, color: AppTheme.textMid)),
              ],
            ),
            const SizedBox(height: 7),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                // An indeterminate bar is honest when the provider hides the
                // file size; a fake percentage would not be.
                value: (p != null && p.total > 0) ? p.fraction : null,
                minHeight: 5,
                backgroundColor: AppTheme.outline,
              ),
            ),
            if (p != null && p.total > 0) ...[
              const SizedBox(height: 5),
              Text(
                '${_human(p.copied)} من ${_human(p.total)}',
                style:
                    const TextStyle(fontSize: 10.8, color: AppTheme.textMid),
              ),
            ],
            const SizedBox(height: 11),
          ],
          SizedBox(
            width: double.infinity,
            height: 48,
            child: FilledButton.icon(
              onPressed: _importing ? null : _importAuto,
              icon: Icon(
                  _importing
                      ? Icons.hourglass_top_rounded
                      : Icons.drive_folder_upload_rounded,
                  size: 20),
              label: Text(
                _importing ? 'جارٍ التهيئة…' : 'استيراد الملفات (تعرّف تلقائي)',
                style: const TextStyle(
                    fontSize: 14.5, fontWeight: FontWeight.w700),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
