import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:markdown_widget/markdown_widget.dart';

Widget markdown(String content) {
  return MarkdownWidget(data: content, shrinkWrap: true);
}

String prettyJson(dynamic data) {
  final encoder = JsonEncoder.withIndent("  ");
  return encoder.convert(data);
}
