import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:grasp/config.dart';
import 'package:grasp/utils.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

final GlobalKey<ScaffoldMessengerState> rootScaffoldMessenger =
    GlobalKey<ScaffoldMessengerState>();

class CustomScrollBehavior extends MaterialScrollBehavior {
  @override
  Set<PointerDeviceKind> get dragDevices => {
    PointerDeviceKind.mouse,
    PointerDeviceKind.touch,
  };
}

void main() {
  runApp(const App());
}

class App extends StatelessWidget {
  const App({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GRASP',
      scrollBehavior: CustomScrollBehavior(),
      scaffoldMessengerKey: rootScaffoldMessenger,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: uniBlue),
        scaffoldBackgroundColor: Colors.white,
        canvasColor: Colors.white,
      ),
      home: const GRASP(),
    );
  }
}

class GRASP extends StatefulWidget {
  const GRASP({super.key});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  @override
  State<GRASP> createState() => _GRASPState();
}

class Past {
  final List<dynamic> questions;
  final List<dynamic> messages;
  final List<dynamic> known;

  Past(this.questions, this.messages, this.known);

  Map<String, dynamic> toJson() {
    return {"questions": questions, "messages": messages, "known": known};
  }
}

class _GRASPState extends State<GRASP> {
  final retry = 10;
  bool initial = true;
  bool running = false;
  bool cancelling = false;

  ScrollController selectionController = ScrollController();
  ScrollController scrollController = ScrollController();
  TextEditingController questionController = TextEditingController();
  FocusNode questionFocus = FocusNode();
  WebSocketChannel? channel;
  Timer? timer;

  int task = 0;
  dynamic lastData;
  dynamic config;
  List<List<dynamic>> histories = [];
  Past? past;
  DateTime lastScrolled = DateTime.now();

  Map<String, bool> knowledgeGraphs = {};

  List<String> get selectedKgs => knowledgeGraphs.entries
      .where((entry) => entry.value)
      .map((entry) => entry.key)
      .toList();

  int get numSelected =>
      knowledgeGraphs.entries.where((entry) => entry.value).length;

  Future<void> connect() async {
    try {
      // reset stuff
      config = null;
      knowledgeGraphs.clear();
      await channel?.sink.close();
      channel = null;

      // open new ws connection
      final newChannel = WebSocketChannel.connect(Uri.parse(wsEndpoint));
      await newChannel.ready;

      // get stuff
      var res = await http.get(Uri.parse(configEndpoint));
      final newConfig = jsonDecode(res.body);
      res = await http.get(Uri.parse(kgEndpoint));
      final newKgs = jsonDecode(res.body).cast<String>() as List<String>;
      assert(newKgs.isNotEmpty);

      // set stuff
      final prefs = await SharedPreferences.getInstance();
      if (initial && prefs.containsKey("lastOutput")) {
        // check past history on initial load
        final lastOutput = prefs.getString("lastOutput");
        final lastData = jsonDecode(lastOutput!);
        past = Past(
          lastData["pastQuestions"],
          lastData["pastMessages"],
          lastData["pastKnown"],
        );
        histories = lastData["histories"].cast<List<dynamic>>();
      }

      if (initial && prefs.containsKey("task")) {
        task = prefs.getInt("task")!;
      }

      final prevSelected = prefs.getStringList("selectedKgs") ?? [];
      config = newConfig;
      for (final kg in newKgs) {
        knowledgeGraphs[kg] = prevSelected.contains(kg);
      }
      if (numSelected == 0) {
        // find wikidata first
        if (knowledgeGraphs.containsKey("wikidata")) {
          knowledgeGraphs["wikidata"] = true;
        } else {
          knowledgeGraphs[newKgs.first] = true;
        }
      }
      channel = newChannel;
    } catch (e) {
      showMessage(
        "Failed to connect to backend. Retrying in $retry seconds.",
        color: uniRed,
      );
      debugPrint("error connecting: $e");
    }
  }

  bool get connected =>
      config != null &&
      knowledgeGraphs.isNotEmpty &&
      channel != null &&
      channel!.closeCode == null;

  void received({bool cancel = false}) {
    channel?.sink.add(jsonEncode({"received": true, "cancel": cancel}));
  }

  void ask(String question) {
    running = true;
    // initialize new history with question
    histories.add([
      {"typ": "question", "question": question},
    ]);
    channel?.sink.add(
      jsonEncode({
        "task": Task.values[task].identifier,
        "question": question,
        "knowledge_graphs": selectedKgs,
        "past": past?.toJson(),
      }),
    );
    setState(() {});
  }

  void cancel() {
    cancelling = true;
    setState(() {});
  }

  Future<void> clear({bool full = false}) async {
    cancelling = false;
    running = false;
    if (full) {
      questionController.text = "";
      histories.clear();
      past = null;
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove("lastOutput");
    } else if (histories.isNotEmpty) {
      histories.removeLast();
    }
    setState(() {});
  }

  void startConnectTimer() {
    timer = Timer.periodic(Duration(seconds: retry), (_) async {
      if (connected) return;

      await connect();
      setState(() {});
    });
  }

  @override
  void initState() {
    super.initState();

    connect().then(
      (_) {
        startConnectTimer();
        initial = false;
        setState(() {});
      },
      onError: (_) {
        startConnectTimer();
        initial = false;
        setState(() {});
      },
    );
  }

  @override
  void dispose() {
    channel?.sink.close();
    questionController.dispose();
    timer?.cancel();
    questionFocus.dispose();
    scrollController.dispose();
    selectionController.dispose();
    super.dispose();
  }

  Widget buildSelection() {
    var children = [
      ActionChip(
        avatar: Icon(Icons.assignment_late_outlined),
        label: Text(Task.values[task].name),
        tooltip: Task.values[task].tooltip,
        visualDensity: VisualDensity.compact,
        onPressed: () async {
          task = (task + 1) % Task.values.length;
          final prefs = await SharedPreferences.getInstance();
          await prefs.setInt("task", task);
          setState(() {});
        },
      ),
    ];
    children.addAll(
      knowledgeGraphs.entries.map((entry) {
        return ActionChip(
          tooltip: entry.value
              ? "Exclude ${entry.key}"
              : "Include ${entry.key}",
          label: Text(
            entry.key,
            style: TextStyle(color: entry.value ? Colors.white : null),
          ),
          backgroundColor: entry.value ? uniBlue : null,
          visualDensity: VisualDensity.compact,
          onPressed: () async {
            if (entry.value && numSelected <= 1) return;
            knowledgeGraphs[entry.key] = !entry.value;
            final prefs = await SharedPreferences.getInstance();
            await prefs.setStringList("selectedKgs", selectedKgs);
            setState(() {});
          },
        );
      }),
    );
    return Wrap(
      crossAxisAlignment: WrapCrossAlignment.center,
      spacing: 8,
      runSpacing: 4,
      alignment: WrapAlignment.center,
      runAlignment: WrapAlignment.center,
      children: children,
    );
  }

  Widget buildTextField() {
    final question = questionController.text.trim();

    final inAction = cancelling || !connected || running;

    return KeyboardListener(
      focusNode: questionFocus,
      onKeyEvent: (event) {
        if (!inAction &&
            question.isNotEmpty &&
            event is KeyDownEvent &&
            event.logicalKey == LogicalKeyboardKey.enter &&
            HardwareKeyboard.instance.isControlPressed) {
          ask(question);
        }
      },
      child: TextField(
        minLines: 1,
        maxLines: 5,
        keyboardType: TextInputType.multiline,
        controller: questionController,
        autofocus: true,
        onChanged: (value) {
          setState(() {});
        },
        decoration: InputDecoration(
          border: OutlineInputBorder(),
          hintText: past == null
              ? "Ask a question..."
              : "Follow up on the previous question...",
          helperText: cancelling
              ? "Cancelling..."
              : "Ctrl + Enter to submit question",
          suffixIcon: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              IconButton(
                tooltip: !connected
                    ? "Disconnected"
                    : cancelling
                    ? "Cancelling"
                    : running
                    ? "Cancel"
                    : "Ask",
                icon: cancelling || !connected
                    ? SizedBox(
                        height: 16,
                        width: 16,
                        child: CircularProgressIndicator(
                          color: connected ? null : uniRed,
                        ),
                      )
                    : running
                    ? Icon(Icons.cancel_outlined, color: uniRed)
                    : Icon(
                        Icons.question_answer_outlined,
                        color: question.isEmpty ? null : uniBlue,
                      ),
                onPressed: cancelling || !connected
                    ? null
                    : running
                    ? () => cancel()
                    : question.isNotEmpty
                    ? () => ask(question)
                    : null,
              ),
              IconButton(
                tooltip: "Reset for new question",
                onPressed: inAction || histories.isEmpty
                    ? null
                    : () async => await clear(full: true),
                icon: Icon(
                  Icons.refresh,
                  color: inAction || histories.isEmpty ? null : uniRed,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget buildCardWithTitle(
    String title,
    Widget content, {
    Color? color,
    Widget? subtitle,
  }) {
    return buildCard(
      Text(
        title,
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.w700,
          fontSize: 18,
        ),
      ),
      content,
      subtitle: subtitle,
    );
  }

  Widget buildCard(Widget title, Widget content, {Widget? subtitle}) {
    return Card(
      margin: EdgeInsets.zero,
      elevation: 1,
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(
        side: BorderSide(color: uniGray),
        borderRadius: BorderRadius.circular(4),
      ),
      color: Colors.white,
      child: Padding(
        padding: EdgeInsets.all(8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [title, Divider(height: 16, thickness: 0.5), content],
        ),
      ),
    );
  }

  Widget buildQuestionItem(String question) {
    return buildCardWithTitle("Question", markdown(question), color: uniGreen);
  }

  Widget buildUnknownItem(dynamic item) {
    return buildCard(
      item["typ"],
      markdown("```json\n${prettyJson(item)}\n```"),
    );
  }

  String fnToMarkdown(dynamic fn) {
    return '''
**${fn["name"]}**

${fn["description"]}

*JSON Schema*
```json
${prettyJson(fn["parameters"])}
```
''';
  }

  Widget buildSystemItem(List<dynamic> functions, String systemMessage) {
    final shape = RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(4),
      side: BorderSide(color: uniGray),
    );

    return buildCardWithTitle(
      "System",
      Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ExpansionTile(
            title: Text("Configuration"),
            tilePadding: EdgeInsets.symmetric(horizontal: 8),
            childrenPadding: EdgeInsets.all(8),
            shape: shape,
            collapsedShape: shape,
            dense: true,
            visualDensity: VisualDensity.compact,
            children: [markdown("```json\n${prettyJson(config)}\n```")],
          ),
          SizedBox(height: 8),
          ExpansionTile(
            title: Text("Functions"),
            tilePadding: EdgeInsets.symmetric(horizontal: 8),
            childrenPadding: EdgeInsets.all(8),
            shape: shape,
            collapsedShape: shape,
            dense: true,
            visualDensity: VisualDensity.compact,
            children: [
              markdown(functions.map((fn) => fnToMarkdown(fn)).join("\n\n")),
            ],
          ),
          SizedBox(height: 8),
          ExpansionTile(
            tilePadding: EdgeInsets.symmetric(horizontal: 8),
            childrenPadding: EdgeInsets.all(8),
            shape: shape,
            collapsedShape: shape,
            dense: true,
            visualDensity: VisualDensity.compact,
            title: Text("GRASP instruction"),
            children: [markdown(systemMessage)],
          ),
        ],
      ),
      color: uniPink,
    );
  }

  Widget buildReasoningItem(String content) {
    return buildCardWithTitle("Reasoning", markdown(content), color: uniBlue);
  }

  Widget buildChip(String key, dynamic value, {Color? color}) {
    return Chip(
      padding: EdgeInsets.all(4),
      label: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Flexible(
            child: Text(
              key,
              style: TextStyle(color: color, fontWeight: FontWeight.w400),
            ),
          ),
          SizedBox(width: 8),
          Flexible(child: Text("$value")),
        ],
      ),
      visualDensity: VisualDensity.compact,
    );
  }

  Iterable<(String, dynamic)> flattenFunctionCallArgs(
    Map<String, dynamic> args, {
    String? prefix,
  }) sync* {
    final pfx = prefix ?? "";
    for (final entry in args.entries) {
      if (entry.value == null) {
        continue;
      } else if (entry.value is Map<String, dynamic>) {
        yield* flattenFunctionCallArgs(
          entry.value,
          prefix: pfx.isEmpty ? entry.key : "$pfx.${entry.key}",
        );
      } else {
        final key = pfx.isEmpty ? entry.key : "$pfx.${entry.key}";
        yield (key, entry.value);
      }
    }
  }

  Widget buildFunctionCallItem(String name, dynamic args, String result) {
    final chips = [buildChip("function", name, color: uniYellow)];
    String? sparql;
    for (final (key, value) in flattenFunctionCallArgs(args)) {
      if (key == "sparql") {
        sparql = value;
        continue;
      }
      chips.add(buildChip(key, value, color: uniYellow));
    }
    return buildCardWithTitle(
      "Function Call",
      Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            crossAxisAlignment: WrapCrossAlignment.center,
            runSpacing: 8,
            spacing: 8,
            children: chips,
          ),
          if (sparql != null) markdown("```sparql\n$sparql\n```"),
          markdown(result),
        ],
      ),
      color: uniYellow,
    );
  }

  Widget buildSparqlQaOutputItem(
    String content,
    String? sparql,
    String? endpoint,
    String? result,
    double elapsed,
  ) {
    final parsed = endpoint != null ? Uri.parse(endpoint) : null;
    return buildCardWithTitle(
      "Output",
      Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          markdown('''$content
          
```sparql
${sparql ?? "No SPARQL query generated."}
```

${result ?? "No SPARQL result available."}
'''),
          Divider(height: 16),
          Wrap(
            alignment: WrapAlignment.start,
            crossAxisAlignment: WrapCrossAlignment.center,
            spacing: 8,
            runSpacing: 8,
            children: [
              Text(
                "Took ${elapsed.toStringAsFixed(2)} seconds",
                style: TextStyle(fontSize: 12),
              ),
              if (sparql != null &&
                  endpoint != null &&
                  parsed != null &&
                  parsed.host == "qlever.cs.uni-freiburg.de")
                TextButton.icon(
                  onPressed: () async {
                    var query =
                        "$parsed?query=${Uri.encodeComponent(sparql)}&exec=true";
                    query = query.replaceFirst("/api", "");
                    await launchUrl(Uri.parse(query));
                  },
                  icon: Icon(Icons.link),
                  label: Text(
                    "Execute on QLever",
                    style: TextStyle(fontSize: 12),
                  ),
                ),
            ],
          ),
        ],
      ),
      color: uniDarkBlue,
    );
  }

  Widget buildGeneralQaOutputItem(String content, double elapsed) {
    return buildCardWithTitle(
      "Output",
      Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          markdown(content),
          Divider(height: 16),
          Text(
            "Took ${elapsed.toStringAsFixed(2)} seconds",
            style: TextStyle(fontSize: 12),
          ),
        ],
      ),
      color: uniDarkBlue,
    );
  }

  Widget buildFeedbackItem(String status, String feedback) {
    return buildCardWithTitle(
      "Feedback",
      Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          buildChip("status", status, color: uniBlue),
          // SizedBox(height: 8),
          markdown(feedback),
        ],
      ),
      color: uniBlue,
    );
  }

  Widget buildHistoryItem(dynamic item) {
    switch (item["typ"] as String) {
      case "question":
        return buildQuestionItem(item["question"]);
      case "system":
        return buildSystemItem(item["functions"], item["system_message"]);
      case "feedback":
        return buildFeedbackItem(item["status"], item["feedback"]);
      case "model":
        return buildReasoningItem(item["content"]);
      case "tool":
        return buildFunctionCallItem(
          item["name"],
          item["args"],
          item["result"],
        );
      case "output":
        final task = item["task"];
        if (task == Task.sparqlQa.identifier) {
          return buildSparqlQaOutputItem(
            item["content"],
            item["sparql"],
            item["endpoint"],
            item["result"],
            item["elapsed"],
          );
        } else if (task == Task.generalQa.identifier) {
          return buildGeneralQaOutputItem(item["content"], item["elapsed"]);
        } else {
          // unknown task
          return buildUnknownItem(item);
        }
      default:
        return buildUnknownItem(item);
    }
  }

  void doDelayed(void Function() fn) {
    Future.delayed(Duration.zero, () => fn());
  }

  void showMessageDelayed(String message, {Color? color}) {
    doDelayed(() => showMessage(message, color: color));
  }

  void showMessage(String message, {Color? color}) {
    rootScaffoldMessenger.currentState?.showSnackBar(
      SnackBar(
        content: Text(message),
        margin: EdgeInsets.all(8),
        behavior: SnackBarBehavior.floating,
        backgroundColor: color,
        duration: Duration(seconds: min(3, retry)),
      ),
    );
  }

  Widget constrainWidth(Widget widget) {
    return ConstrainedBox(
      constraints: BoxConstraints(maxWidth: 800),
      child: widget,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: initial
          ? Center(child: CircularProgressIndicator())
          : StreamBuilder(
              stream: channel?.stream,
              builder: (_, data) {
                if (data.hasError) {
                  showMessageDelayed(
                    "Unknown error: ${data.error}",
                    color: uniRed,
                  );
                } else if (data.hasData && data.data != lastData) {
                  lastData = data.data;
                  final json = jsonDecode(data.data);
                  final hasTyp = json.containsKey("typ");
                  if (!hasTyp && json.containsKey("error")) {
                    showMessageDelayed(json["error"], color: uniRed);
                  } else if (!hasTyp && json.containsKey("cancelled")) {
                    doDelayed(() => clear());
                  } else if (hasTyp) {
                    received(cancel: cancelling);
                    histories.last.add(json);
                    if (json["typ"] == "output") {
                      questionController.text = "";
                      cancelling = false;
                      running = false;
                      past = Past(
                        json["questions"],
                        json["messages"],
                        json["known"],
                      );
                      SharedPreferences.getInstance().then(
                        (prefs) => prefs.setString(
                          "lastOutput",
                          jsonEncode({
                            "pastQuestions": past!.questions,
                            "pastMessages": past!.messages,
                            "pastKnown": past!.known,
                            "histories": histories,
                          }),
                        ),
                      );
                    }
                  }
                }
                final canScroll =
                    scrollController.hasClients &&
                    scrollController.position.pixels <=
                        scrollController.position.maxScrollExtent -
                            10; // some small tolerance

                final items = histories.expand((h) => h).toList();
                return Center(
                  child: Padding(
                    padding: EdgeInsets.all(8),
                    child: constrainWidth(
                      Column(
                        mainAxisAlignment: histories.isEmpty
                            ? MainAxisAlignment.center
                            : MainAxisAlignment.spaceBetween,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        mainAxisSize: MainAxisSize.max,
                        children: [
                          if (histories.isNotEmpty) ...[
                            Expanded(
                              child: Stack(
                                alignment: AlignmentDirectional.bottomCenter,
                                children: [
                                  NotificationListener<ScrollNotification>(
                                    onNotification: (_) {
                                      final now = DateTime.now();
                                      final diff = now.difference(lastScrolled);
                                      if (diff.inMilliseconds > 200) {
                                        lastScrolled = now;
                                        setState(() {});
                                      }
                                      return false;
                                    },
                                    child: ListView.separated(
                                      padding: EdgeInsets.zero,
                                      controller: scrollController,
                                      itemCount: items.length,
                                      itemBuilder: (_, i) =>
                                          buildHistoryItem(items[i]),
                                      separatorBuilder: (_, i) =>
                                          SizedBox(height: 8),
                                    ),
                                  ),
                                  if (canScroll)
                                    FloatingActionButton(
                                      mini: true,
                                      backgroundColor: Colors.white,
                                      shape: RoundedRectangleBorder(
                                        side: BorderSide(color: uniGray),
                                        borderRadius: BorderRadius.circular(
                                          100,
                                        ),
                                      ),
                                      tooltip: "Scroll to bottom",
                                      onPressed: () async {
                                        await scrollController.animateTo(
                                          scrollController
                                              .position
                                              .maxScrollExtent,
                                          duration: Duration(milliseconds: 200),
                                          curve: Curves.easeInOut,
                                        );
                                        setState(() {});
                                      },
                                      child: Icon(
                                        Icons.arrow_downward_outlined,
                                      ),
                                    ),
                                ],
                              ),
                            ),
                            SizedBox(height: 8),
                          ],
                          buildTextField(),
                          SizedBox(height: 8),
                          buildSelection(),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
    );
  }
}
