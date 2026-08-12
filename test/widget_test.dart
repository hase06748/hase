import 'package:flutter_test/flutter_test.dart';
import 'package:photo_enhancer/main.dart';

void main() {
  testWidgets('App builds without crashing', (WidgetTester tester) async {
    await tester.pumpWidget(const PhotoEnhancerApp());
    expect(find.text('محسّن الصور'), findsOneWidget);
  });
}
