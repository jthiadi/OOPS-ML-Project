import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:fl_chart/fl_chart.dart';

void main() {
  runApp(const LeftItemApp());
}

// CHANGE this if your API runs on different host/port
const String apiBase = 'http://localhost:8000';

class LeftItemApp extends StatelessWidget {
  const LeftItemApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Library Left-Item Dashboard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xfff2f4fb),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xff4b5dd8),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const DashboardPage(),
    );
  }
}

// ----------------- DATA MODELS -----------------
class DailyStat {
  final DateTime date;
  final int count;

  DailyStat({required this.date, required this.count});

  factory DailyStat.fromJson(Map<String, dynamic> json) {
    return DailyStat(
      date: DateTime.parse(json['date'] as String),
      count: json['count'] as int,
    );
  }
}

class CategoryStat {
  final String itemName;
  final int count;

  CategoryStat({required this.itemName, required this.count});

  factory CategoryStat.fromJson(Map<String, dynamic> json) {
    return CategoryStat(
      itemName: json['item_name'] as String,
      count: json['count'] as int,
    );
  }
}

class LeftItem {
  final int id;
  final int objId;
  final String itemName;
  final String ownerSide;
  final String tableId;
  final String capturedAt;
  final String? grokDesc;

  LeftItem({
    required this.id,
    required this.objId,
    required this.itemName,
    required this.ownerSide,
    required this.tableId,
    required this.capturedAt,
    this.grokDesc,
  });

  factory LeftItem.fromJson(Map<String, dynamic> json) {
    return LeftItem(
      id: json['id'] as int,
      objId: json['obj_id'] as int,
      itemName: json['item_name'] as String,
      ownerSide: json['owner_side'] as String,
      tableId: json['table_id'] as String? ?? '',
      capturedAt: json['captured_at'] as String,
      grokDesc: json['grok_desc'] as String?,
    );
  }
}

// ----------------- API HELPERS -----------------
Future<List<DailyStat>> fetchDailyStats() async {
  final res = await http.get(Uri.parse('$apiBase/stats/30days'));
  if (res.statusCode != 200) {
    throw Exception('Failed to fetch daily stats: ${res.statusCode}');
  }
  final List data = jsonDecode(res.body);
  return data
      .map((e) => DailyStat.fromJson(e as Map<String, dynamic>))
      .toList();
}

Future<List<CategoryStat>> fetchCategoryStats() async {
  final res = await http.get(Uri.parse('$apiBase/stats/category'));
  if (res.statusCode != 200) {
    throw Exception('Failed to fetch category stats: ${res.statusCode}');
  }
  final List data = jsonDecode(res.body);
  return data
      .map((e) => CategoryStat.fromJson(e as Map<String, dynamic>))
      .toList();
}

Future<List<LeftItem>> fetchRecentItems() async {
  final res = await http
      .get(Uri.parse('$apiBase/items/recent?days=30&limit=200')); // more rows
  if (res.statusCode != 200) {
    throw Exception('Failed to fetch recent items: ${res.statusCode}');
  }
  final List data = jsonDecode(res.body);
  return data
      .map((e) => LeftItem.fromJson(e as Map<String, dynamic>))
      .toList();
}

// live snapshot frame
Future<Uint8List?> fetchLiveFrame() async {
  final res = await http.get(
    Uri.parse('$apiBase/live-frame?ts=${DateTime.now().millisecondsSinceEpoch}'),
  );
  if (res.statusCode != 200) {
    return null;
  }
  return res.bodyBytes;
}

// per-item image for preview panel
Future<Uint8List?> fetchItemImage(int id) async {
  final res = await http.get(Uri.parse('$apiBase/item-image/$id'));
  if (res.statusCode != 200) {
    return null;
  }
  return res.bodyBytes;
}

// NEW: call backend to generate AI description and save to DB
Future<String?> describeItem(int id) async {
  final res = await http.post(Uri.parse('$apiBase/items/$id/describe'));
  if (res.statusCode != 200) {
    return null;
  }
  final Map<String, dynamic> data = jsonDecode(res.body);
  return data['description'] as String?;
}

// ----------------- EXTRA ENUMS FOR STATUS -----------------
enum SeatStatus { seated, temporaryLeave, leftBehind }
enum ItemStatus { available, retrieved }

// ----------------- DASHBOARD PAGE -----------------
class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  late Future<List<DailyStat>> _dailyFuture;
  late Future<List<CategoryStat>> _catFuture;
  late Future<List<LeftItem>> _recentFuture;

  Uint8List? _liveFrameBytes;
  bool _loadingFrame = false;
  bool _detectorRunning = false;
  bool _sidebarOpen = true;

  Timer? _frameTimer;

  // filters for table
  final TextEditingController _searchController = TextEditingController();
  DateTimeRange? _dateRange;

  // pagination
  final int _rowsPerPage = 10;
  int _currentPage = 0;

  // selected item preview
  LeftItem? _selectedItem;
  Uint8List? _selectedItemImage;
  bool _loadingItemImage = false;

  // AI description state
  String? _selectedDescription;
  bool _describeLoading = false;

  // items that we know have images (after successful load)
  final List<LeftItem> _itemsWithImage = [];

  // seat status (for live snapshot bullets)
  SeatStatus _leftSeatStatus = SeatStatus.seated;
  SeatStatus _rightSeatStatus = SeatStatus.temporaryLeave;

  // item status bullets (available / retrieved)
  ItemStatus _selectedItemStatus = ItemStatus.available;

  @override
  void initState() {
    super.initState();

    _dailyFuture = fetchDailyStats();
    _catFuture = fetchCategoryStats();
    _recentFuture = fetchRecentItems();

    _loadFrame(); // once at start
  }

  @override
  void dispose() {
    _frameTimer?.cancel();
    _searchController.dispose();
    super.dispose();
  }

  // ---------- AI DESCRIBE: SINGLE VERSION ----------
  Future<void> _describeSelectedItem() async {
    final item = _selectedItem;
    if (item == null) return;

    setState(() {
      _describeLoading = true;
    });

    try {
      final desc = await describeItem(item.id);

      if (!mounted) return;
      setState(() {
        _selectedDescription = desc;

        final updated = LeftItem(
          id: item.id,
          objId: item.objId,
          itemName: item.itemName,
          ownerSide: item.ownerSide,
          tableId: item.tableId,
          capturedAt: item.capturedAt,
          grokDesc: desc,
        );
        _selectedItem = updated;

        final idx = _itemsWithImage.indexWhere((it) => it.id == item.id);
        if (idx != -1) {
          _itemsWithImage[idx] = updated;
        }
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to describe: $e')),
      );
    } finally {
      if (mounted) {
        setState(() {
          _describeLoading = false;
        });
      }
    }
  }

  // -------- selected item image ----------
  Future<void> _loadSelectedItemImage() async {
    final item = _selectedItem;
    if (item == null) return;

    setState(() {
      _loadingItemImage = true;
      _selectedItemImage = null;
    });

    try {
      final bytes = await fetchItemImage(item.id);
      setState(() {
        _selectedItemImage = bytes; // can be null if not found
        if (bytes != null && bytes.isNotEmpty) {
          final exists = _itemsWithImage.any((it) => it.id == item.id);
          if (!exists) {
            _itemsWithImage.add(item);
          }
        }
      });
    } catch (_) {
      setState(() {
        _selectedItemImage = null;
      });
    } finally {
      if (mounted) {
        setState(() {
          _loadingItemImage = false;
        });
      }
    }
  }

  // -------- detector controls ----------
  Future<void> _startDetector() async {
    try {
      final res = await http.post(Uri.parse('$apiBase/detect/start'));
      if (!mounted) return;

      if (res.statusCode == 200) {
        setState(() {
          _detectorRunning = true;
        });

        // polling frame tiap 3 detik
        _frameTimer?.cancel();
        _frameTimer = Timer.periodic(const Duration(seconds: 3), (_) {
          _loadFrame();
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Detector started')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to start detector: ${res.statusCode}')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to start detector: $e')),
      );
    }
  }

  Future<void> _stopDetector() async {
    try {
      final res = await http.post(Uri.parse('$apiBase/detect/stop'));
      if (!mounted) return;

      setState(() {
        _detectorRunning = false;
      });

      // STOP polling frame
      _frameTimer?.cancel();
      _frameTimer = null;

      if (res.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Detector stopped')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to stop detector: ${res.statusCode}')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to stop detector: $e')),
      );
    }
  }

  Future<void> _loadFrame() async {
    setState(() => _loadingFrame = true);
    final bytes = await fetchLiveFrame();
    setState(() {
      _liveFrameBytes = bytes;
      _loadingFrame = false;
    });
  }

  // table filter logic (search + date range + sort newest -> oldest)
  List<LeftItem> _filterItems(List<LeftItem> items) {
    final query = _searchController.text.trim().toLowerCase();

    Iterable<LeftItem> result = items;

    if (query.isNotEmpty) {
      result = result.where((item) {
        final s = [
          item.itemName,
          item.ownerSide,
          item.tableId,
          item.capturedAt,
          item.grokDesc ?? '',
        ].join(' ').toLowerCase();
        return s.contains(query);
      });
    }

    if (_dateRange != null) {
      final start = _dateRange!.start;
      final end = _dateRange!.end.add(const Duration(days: 1));
      result = result.where((item) {
        DateTime dt;
        try {
          dt = DateTime.parse(item.capturedAt);
        } catch (_) {
          return true; // if parse fails, keep it
        }
        return (dt.isAtSameMomentAs(start) || dt.isAfter(start)) &&
            dt.isBefore(end);
      });
    }

    final list = result.toList();

    // sort newest -> oldest
    list.sort((a, b) {
      try {
        final da = DateTime.parse(a.capturedAt);
        final db = DateTime.parse(b.capturedAt);
        return db.compareTo(da);
      } catch (_) {
        return 0;
      }
    });

    return list;
  }

  Future<void> _pickDateRange() async {
    final now = DateTime.now();
    final result = await showDateRangePicker(
      context: context,
      firstDate: DateTime(now.year - 1),
      lastDate: DateTime(now.year + 1),
      initialDateRange: _dateRange ??
          DateTimeRange(
            start: now.subtract(const Duration(days: 7)),
            end: now,
          ),
    );
    if (result != null) {
      setState(() {
        _dateRange = result;
        _currentPage = 0; // reset to first page
      });
    }
  }

  Widget _buildItemsWithImageList() {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
      ),
      child: _itemsWithImage.isEmpty
          ? const Center(
              child: Text(
                "Items with images will appear here",
                style: TextStyle(fontSize: 11, color: Colors.grey),
              ),
            )
          : ListView.builder(
              itemCount: _itemsWithImage.length,
              itemBuilder: (context, index) {
                final it = _itemsWithImage[index];
                final desc = (it.grokDesc ?? '').trim();

                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text(
                    desc.isNotEmpty
                        ? "#${it.id} • $desc"
                        : "#${it.id} • ${it.itemName}",
                    style: const TextStyle(fontSize: 11),
                  ),
                );
              },
            ),
    );
  }

  Color _seatStatusColor(SeatStatus status) {
    switch (status) {
      case SeatStatus.seated:
        return Colors.green;
      case SeatStatus.temporaryLeave:
        return Colors.red;
      case SeatStatus.leftBehind:
        return Colors.grey;
    }
  }

  Widget _seatStatusBullet(String label, SeatStatus status) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            color: _seatStatusColor(status),
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 6),
        Flexible(
          child: Text(
            label,
            style: const TextStyle(fontSize: 11),
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }

  Widget _itemStatusPill(String label, bool active) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: active ? Colors.green.withOpacity(0.12) : Colors.grey.shade200,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(
          color: active ? Colors.green : Colors.grey.shade400,
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: active ? Colors.green : Colors.grey.shade500,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: active ? FontWeight.w600 : FontWeight.w400,
              color: active ? Colors.green.shade800 : Colors.grey.shade700,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSelectedItemPreview() {
    if (_selectedItem == null) {
      // nothing selected yet
      return Container(
        height: 260,
        decoration: BoxDecoration(
          color: Colors.grey.shade100,
          borderRadius: BorderRadius.circular(16),
        ),
        alignment: Alignment.center,
        child: const Text(
          "Select a row to preview the item",
          style: TextStyle(fontSize: 12, color: Colors.grey),
          textAlign: TextAlign.center,
        ),
      );
    }

    final item = _selectedItem!;

    Widget imageArea;
    if (_loadingItemImage) {
      imageArea = const Center(child: CircularProgressIndicator());
    } else if (_selectedItemImage != null) {
      imageArea = ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.memory(
          _selectedItemImage!,
          fit: BoxFit.cover,
          width: double.infinity,
          height: 170,
        ),
      );
    } else {
      // attempted but not found
      imageArea = Container(
        width: double.infinity,
        height: 170,
        decoration: BoxDecoration(
          color: Colors.grey.withOpacity(0.4),
          borderRadius: BorderRadius.circular(12),
        ),
        alignment: Alignment.center,
        child: const Text(
          "No image found",
          style: TextStyle(color: Colors.white70, fontSize: 12),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xfff5f6ff),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            "Item #${item.id}",
            style: const TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            "${item.itemName} • ${item.tableId} • ${item.ownerSide} side",
            style: const TextStyle(fontSize: 12, color: Colors.black54),
          ),
          const SizedBox(height: 8),

          // Describe button
          Row(
            children: [
              OutlinedButton.icon(
                onPressed: _describeLoading ? null : _describeSelectedItem,
                icon: _describeLoading
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.auto_awesome, size: 16),
                label: const Text("Describe", style: TextStyle(fontSize: 12)),
              ),
            ],
          ),

          const SizedBox(height: 8),
          imageArea,
          const SizedBox(height: 8),

          // AI description box
          Container(
            height: 80,
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
            ),
            child: _describeLoading
                ? const Center(
                    child: SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  )
                : (_selectedDescription != null &&
                        _selectedDescription!.isNotEmpty)
                    ? Text(
                        _selectedDescription!,
                        style: const TextStyle(
                          fontSize: 11,
                          color: Colors.black87,
                        ),
                      )
                    : const Center(
                        child: Text(
                          "No AI description yet.\nPress \"Describe\" to generate.",
                          style: TextStyle(fontSize: 11, color: Colors.grey),
                          textAlign: TextAlign.center,
                        ),
                      ),
          ),

          const SizedBox(height: 8),

          // Item status pills: Available / Retrieved
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _itemStatusPill(
                "Available",
                _selectedItemStatus == ItemStatus.available,
              ),
              _itemStatusPill(
                "Retrieved",
                _selectedItemStatus == ItemStatus.retrieved,
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          AnimatedContainer(
            width: _sidebarOpen ? 220 : 0,
            duration: const Duration(milliseconds: 250),
            child: _sidebarOpen ? _buildSidebar(context) : null,
          ),
          Expanded(
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildHeader(),
                      const SizedBox(height: 16),
                      _buildTopRow(),    // pie + line chart
                      const SizedBox(height: 16),
                      _buildMiddleRow(), // live snapshot + seat bullets
                      const SizedBox(height: 16),
                      _buildBottomRow(), // table + preview
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ----------------- LAYOUT PIECES -----------------
  Widget _buildSidebar(BuildContext context) {
    return Container(
      color: const Color(0xffe5e9ff),
      child: Column(
        children: [
          const SizedBox(height: 24),
          const Text(
            "Library Analytics",
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Color(0xff273469),
            ),
          ),
          const SizedBox(height: 40),
          _sideItem(
            Icons.dashboard,
            "Dashboard",
            selected: true,
          ),
          const Spacer(),
        ],
      ),
    );
  }

  Widget _sideItem(IconData icon, String label,
      {bool selected = false, VoidCallback? onTap}) {
    return InkWell(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 12),
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
        decoration: BoxDecoration(
          color: selected ? const Color(0xff4b5dd8) : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            Icon(
              icon,
              size: 20,
              color: selected ? Colors.white : const Color(0xff4b5dd8),
            ),
            const SizedBox(width: 12),
            Text(
              label,
              style: TextStyle(
                color: selected ? Colors.white : const Color(0xff273469),
                fontWeight: selected ? FontWeight.bold : FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      children: [
        IconButton(
          icon: const Icon(Icons.menu),
          onPressed: () {
            setState(() {
              _sidebarOpen = !_sidebarOpen;
            });
          },
        ),
        const SizedBox(width: 8),
        // make title flexible so it doesn't overflow
        Expanded(
          child: Text(
            "Rental rate / Left-item dashboard",
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Color(0xff273469),
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        const SizedBox(width: 8),
        ElevatedButton.icon(
          onPressed: () {
            setState(() {
              _dailyFuture = fetchDailyStats();
              _catFuture = fetchCategoryStats();
              _recentFuture = fetchRecentItems();
              _currentPage = 0;
            });
          },
          icon: const Icon(Icons.refresh),
          label: const Text("Refresh data"),
        ),
      ],
    );
  }

  // FIRST ROW: Categories + Line chart
  Widget _buildTopRow() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: _card(
            title: "Categories (30 days)",
            child: FutureBuilder<List<CategoryStat>>(
              future: _catFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }
                if (snapshot.hasError) {
                  return Center(child: Text("Error: ${snapshot.error}"));
                }
                final cats = snapshot.data ?? [];
                if (cats.isEmpty) {
                  return const Center(child: Text("No category data."));
                }
                return _CategoryPieChart(stats: cats);
              },
            ),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: _card(
            title: "Left-item count (last 30 days)",
            child: FutureBuilder<List<DailyStat>>(
              future: _dailyFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }
                if (snapshot.hasError) {
                  return Center(child: Text("Error: ${snapshot.error}"));
                }
                final stats = snapshot.data ?? [];
                if (stats.isEmpty) {
                  return const Center(child: Text("No data for last 30 days."));
                }
                return _DailyLineChart(stats: stats);
              },
            ),
          ),
        ),
      ],
    );
  }

  // SECOND ROW: live snapshot full-width with seat bullets
  Widget _buildMiddleRow() {
    return _card(
      title: "Live snapshot",
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // seat bullets
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _seatStatusBullet("Left seat", _leftSeatStatus),
              _seatStatusBullet("Right seat", _rightSeatStatus),
            ],
          ),
          const SizedBox(height: 8),
          if (_loadingFrame)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Center(child: CircularProgressIndicator()),
            )
          else if (_liveFrameBytes != null)
            SizedBox(
              height: 220,
              child: _buildLiveSnapshotImage(),
            )
          else
            const Padding(
              padding: EdgeInsets.all(16),
              child: Text("No frame yet."),
            ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 12,
            runSpacing: 8,
            alignment: WrapAlignment.start,
            children: [
              OutlinedButton.icon(
                onPressed: _startDetector,
                icon: const Icon(Icons.play_arrow),
                label: const Text("Start"),
              ),
              OutlinedButton.icon(
                onPressed: _stopDetector,
                icon: const Icon(Icons.stop),
                label: const Text("Stop"),
              ),
              OutlinedButton.icon(
                onPressed: _loadFrame,
                icon: const Icon(Icons.refresh),
                label: const Text("Refresh"),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // make snapshot dim when detector not running + avoid flicker
  Widget _buildLiveSnapshotImage() {
    final img = ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Image.memory(
        _liveFrameBytes!,
        fit: BoxFit.cover,
        width: double.infinity,
        gaplessPlayback: true,
        key: ValueKey(_liveFrameBytes!.length), // no flicker
      ),
    );

    if (_detectorRunning) {
      return img;
    } else {
      return ColorFiltered(
        colorFilter: ColorFilter.mode(
          Colors.white.withOpacity(0.5),
          BlendMode.srcATop,
        ),
        child: img,
      );
    }
  }

  // THIRD ROW: table + filters + pagination + preview
  Widget _buildBottomRow() {
    return _card(
      title: "Recent left items (last 30 days)",
      child: FutureBuilder<List<LeftItem>>(
        future: _recentFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text("Error: ${snapshot.error}"));
          }
          final items = snapshot.data ?? [];
          if (items.isEmpty) {
            return const Center(child: Text("No items yet."));
          }

          final filteredItems = _filterItems(items);
          final totalItems = filteredItems.length;
          final totalPages =
              (totalItems / _rowsPerPage).ceil().clamp(1, 1000000);
          final currentPage =
              _currentPage.clamp(0, totalPages - 1); // safe page index

          final startIndex = currentPage * _rowsPerPage;
          final endIndex = (startIndex + _rowsPerPage > totalItems)
              ? totalItems
              : startIndex + _rowsPerPage;
          final pageItems = filteredItems.sublist(startIndex, endIndex);

          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // --- search + date range row ---
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    SizedBox(
                      width: 300,
                      child: TextField(
                        controller: _searchController,
                        decoration: const InputDecoration(
                          hintText: 'Search by item / owner / table / date...',
                          prefixIcon: Icon(Icons.search),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.all(Radius.circular(12)),
                          ),
                          isDense: true,
                        ),
                        onChanged: (_) {
                          setState(() {
                            _currentPage = 0;
                          });
                        },
                      ),
                    ),
                    const SizedBox(width: 12),
                    OutlinedButton.icon(
                      onPressed: _pickDateRange,
                      icon: const Icon(Icons.date_range),
                      label: const Text("Date range"),
                    ),
                    if (_dateRange != null) ...[
                      const SizedBox(width: 8),
                      TextButton(
                        onPressed: () {
                          setState(() {
                            _dateRange = null;
                            _currentPage = 0;
                          });
                        },
                        child: const Text("Clear"),
                      ),
                    ],
                  ],
                ),
              ),
              if (_dateRange != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4.0),
                  child: Text(
                    "Showing: ${_dateRange!.start.toString().split(' ').first} "
                    "→ ${_dateRange!.end.toString().split(' ').first}",
                    style: const TextStyle(fontSize: 11, color: Colors.grey),
                  ),
                ),

              const SizedBox(height: 12),

              // --- table + preview side by side ---
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 3,
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: DataTable(
                        showCheckboxColumn: false,
                        columnSpacing: 32,
                        columns: const [
                          DataColumn(label: Text("ID")),
                          DataColumn(label: Text("Item")),
                          DataColumn(label: Text("Owner side")),
                          DataColumn(
                            label: SizedBox(
                              width: 50,
                              child: Text("Table"),
                            ),
                          ),
                          DataColumn(label: Text("Captured at")),
                        ],
                        rows: pageItems
                            .map(
                              (e) => DataRow(
                                selected: _selectedItem?.id == e.id,
                                onSelectChanged: (selected) {
                                  if (selected == true) {
                                    setState(() {
                                      _selectedItem = e;
                                      _selectedItemImage = null;
                                      _loadingItemImage = true;
                                      _selectedDescription = e.grokDesc;
                                      _describeLoading = false;
                                      _selectedItemStatus =
                                          ItemStatus.available; // default
                                    });
                                    _loadSelectedItemImage();
                                  }
                                },
                                cells: [
                                  DataCell(Text(e.id.toString())),
                                  DataCell(Text(e.itemName)),
                                  DataCell(Text(e.ownerSide)),
                                  DataCell(
                                    SizedBox(
                                      width: 50,
                                      child: Text(e.tableId),
                                    ),
                                  ),
                                  DataCell(Text(e.capturedAt)),
                                ],
                              ),
                            )
                            .toList(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Flexible(
                    flex: 2,
                    child: _buildSelectedItemPreview(),
                  ),
                ],
              ),

              const SizedBox(height: 8),

              // --- pagination controls ---
              Row(
                children: [
                  Text(
                    "Showing ${startIndex + 1}–$endIndex of $totalItems "
                    "(Page ${currentPage + 1} / $totalPages)",
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                  const Spacer(),
                  TextButton.icon(
                    onPressed: currentPage > 0
                        ? () {
                            setState(() {
                              _currentPage = currentPage - 1;
                            });
                          }
                        : null,
                    icon: const Icon(Icons.chevron_left),
                    label: const Text("Prev"),
                  ),
                  const SizedBox(width: 8),
                  TextButton.icon(
                    onPressed: currentPage < totalPages - 1
                        ? () {
                            setState(() {
                              _currentPage = currentPage + 1;
                            });
                          }
                        : null,
                    icon: const Icon(Icons.chevron_right),
                    label: const Text("Next"),
                  ),
                ],
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _card({required Widget child, String? title}) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: const [
          BoxShadow(
            color: Color(0x14000000),
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (title != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Color(0xff273469),
                ),
              ),
            ),
          child,
        ],
      ),
    );
  }
}

// ----------------- CHART WIDGETS -----------------

class _DailyLineChart extends StatelessWidget {
  final List<DailyStat> stats;

  const _DailyLineChart({required this.stats});

  @override
  Widget build(BuildContext context) {
    final spots = <FlSpot>[];
    for (int i = 0; i < stats.length; i++) {
      spots.add(FlSpot(i.toDouble(), stats[i].count.toDouble()));
    }

    final maxCount =
        stats.fold<int>(0, (prev, e) => e.count > prev ? e.count : prev);
    final yInterval = maxCount <= 5 ? 1.0 : (maxCount / 4).ceilToDouble();

    return SizedBox(
      height: 260,
      child: LineChart(
        LineChartData(
          minY: 0,
          maxY: (maxCount + yInterval).toDouble(),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 32,
                interval: yInterval,
                getTitlesWidget: (value, meta) {
                  return Text(
                    value.toInt().toString(),
                    style: const TextStyle(fontSize: 10),
                  );
                },
              ),
            ),
            rightTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            topTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                interval: (stats.length / 6).ceilToDouble().clamp(1, 999),
                getTitlesWidget: (val, meta) {
                  final idx = val.toInt();
                  if (idx < 0 || idx >= stats.length) {
                    return const SizedBox.shrink();
                  }
                  final d = stats[idx].date;
                  return Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Transform.rotate(
                      angle: -0.6, // rotate ~ -35deg
                      child: Text(
                        "${d.month}/${d.day}",
                        style: const TextStyle(fontSize: 9),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
          gridData: FlGridData(show: true),
          borderData: FlBorderData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              barWidth: 3,
              dotData: FlDotData(show: true),
              belowBarData: BarAreaData(
                show: true,
                color: Theme.of(context)
                    .colorScheme
                    .primary
                    .withOpacity(0.15),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// interactive pie chart with hover/tap highlight
class _CategoryPieChart extends StatefulWidget {
  final List<CategoryStat> stats;

  const _CategoryPieChart({required this.stats});

  @override
  State<_CategoryPieChart> createState() => _CategoryPieChartState();
}

class _CategoryPieChartState extends State<_CategoryPieChart> {
  int? touchedIndex;

  @override
  Widget build(BuildContext context) {
    final stats = widget.stats;
    final total = stats.fold<int>(0, (sum, s) => sum + s.count);
    if (total == 0) {
      return const Center(child: Text("No items"));
    }

    final palette = <Color>[
      const Color(0xff4b82d8),
      const Color(0xff48a9a6),
      const Color(0xfff29e4c),
      const Color(0xffe26d5c),
      const Color(0xff8d6cab),
      const Color(0xff5fb49c),
    ];

    final sections = <PieChartSectionData>[];
    for (int i = 0; i < stats.length; i++) {
      final s = stats[i];
      final value = s.count.toDouble();
      final baseColor = palette[i % palette.length];
      final isTouched = i == touchedIndex;
      final displayColor =
          Color.lerp(baseColor, Colors.white, isTouched ? 0.25 : 0.0)!;

      sections.add(
        PieChartSectionData(
          color: displayColor,
          value: value,
          title: '',
          radius: isTouched ? 70 : 60,
        ),
      );
    }

    CategoryStat? current;
    double? currentPct;
    if (touchedIndex != null &&
        touchedIndex! >= 0 &&
        touchedIndex! < stats.length) {
      current = stats[touchedIndex!];
      currentPct = 100 * current.count / total;
    }

    return Column(
      children: [
        SizedBox(
          height: 220,
          child: PieChart(
            PieChartData(
              sections: sections,
              sectionsSpace: 2,
              centerSpaceRadius: 40,
              pieTouchData: PieTouchData(
                touchCallback: (event, response) {
                  setState(() {
                    if (!event.isInterestedForInteractions ||
                        response == null ||
                        response.touchedSection == null) {
                      touchedIndex = null;
                    } else {
                      touchedIndex =
                          response.touchedSection!.touchedSectionIndex;
                    }
                  });
                },
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        if (current != null)
          Text(
            "${current.itemName} • ${current.count} items • ${currentPct!.toStringAsFixed(0)}%",
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
          )
        else
          const Text(
            "Hover / tap a slice to see details",
            style: TextStyle(fontSize: 11, color: Colors.grey),
          ),
      ],
    );
  }
}
