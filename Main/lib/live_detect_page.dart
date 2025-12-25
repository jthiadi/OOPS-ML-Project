// live_detect_page.dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'main.dart'; // ← use apiBase from main.dart, so delete local const

class LiveDetectPage extends StatefulWidget {
  const LiveDetectPage({super.key});

  @override
  State<LiveDetectPage> createState() => _LiveDetectPageState();
}

class _LiveDetectPageState extends State<LiveDetectPage> {
  Uint8List? _frame;
  List<String> _logs = [];
  bool _initialising = true;

  @override
  void initState() {
    super.initState();
    _startDetectorAndLoop();
  }

  Future<void> _startDetectorAndLoop() async {
    await http.post(Uri.parse('$apiBase/detect/start'));  // ok if already running
    _refreshLoop();
  }

  Future<void> _refreshLoop() async {
    while (mounted) {
      await Future.wait([_loadFrame(), _loadLogs()]);
      if (_initialising) {
        setState(() => _initialising = false);
      }
      await Future.delayed(const Duration(milliseconds: 500));
    }
  }

  Future<void> _loadFrame() async {
    try {
      final res = await http.get(
        Uri.parse('$apiBase/live-frame?ts=${DateTime.now().millisecondsSinceEpoch}'),
      );
      if (res.statusCode == 200) {
        setState(() {
          _frame = res.bodyBytes;
        });
      }
    } catch (_) {}
  }

  Future<void> _loadLogs() async {
    try {
      final res = await http.get(Uri.parse('$apiBase/logs/live?lines=200'));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        final List<dynamic> lines = data['lines'] ?? [];
        setState(() {
          _logs = lines.cast<String>();
        });
      }
    } catch (_) {}
  }

  Future<void> _stopDetector() async {
    await http.post(Uri.parse('$apiBase/detect/stop'));
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xfff2f4fb),
      appBar: AppBar(
        title: const Text("Live detect"),
        actions: [
          IconButton(
            icon: const Icon(Icons.stop_circle, color: Colors.red),
            tooltip: "Stop detector",
            onPressed: _stopDetector,
          ),
        ],
      ),
      body: Row(
        children: [
          // LEFT: camera
          Expanded(
            flex: 3,
            child: Center(
              child: Container(
                margin: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Center(
                  child: _frame == null
                      ? (_initialising
                          ? const CircularProgressIndicator()
                          : const Text(
                              "No frame yet.\nPastikan main.py sudah menyimpan live_frame.jpg",
                              textAlign: TextAlign.center,
                            ))
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: Image.memory(
                            _frame!,
                            fit: BoxFit.contain,
                            width: double.infinity,
                            height: double.infinity,
                            gaplessPlayback: true,   // sedikit mengurangi flicker
                          ),
                        ),
                ),
              ),
            ),
          ),
          // RIGHT: logs ...
        ],
      ),
    );
  }
}
