import 'package:flutter/material.dart';

// change this config to suit your setup
const hostAndPort = "localhost:8000";
const secure = false;
const baseURL = "";

// do not change that
enum Task { sparqlQa, generalQa }

extension TaskExtension on Task {
  String get identifier {
    switch (this) {
      case Task.sparqlQa:
        return "sparql-qa";
      case Task.generalQa:
        return "general-qa";
    }
  }

  String get name {
    switch (this) {
      case Task.sparqlQa:
        return "SPARQL QA";
      case Task.generalQa:
        return "General QA";
    }
  }

  String get tooltip {
    switch (this) {
      case Task.sparqlQa:
        return "Answer questions by generating a corresponding SPARQL query "
            "over one or more knowledge graphs";
      case Task.generalQa:
        return "Answer questions by retrieving relevant information from "
            "knowledge graphs";
    }
  }
}

const wsPath = "/live";
const configPath = "/config";
const kgPath = "/knowledge_graphs";

get backendEndpoint => "http${secure ? "s" : ""}://$hostAndPort$baseURL";

get configEndpoint => backendEndpoint + configPath;

get kgEndpoint => backendEndpoint + kgPath;

get wsEndpoint => "ws${secure ? "s" : ""}://$hostAndPort$baseURL$wsPath";

const uniBlue = Color.fromRGBO(52, 74, 154, 1);
const uniDarkBlue = Color.fromRGBO(0, 1, 73, 1);
const uniRed = Color.fromRGBO(193, 0, 42, 1);
const uniGray = Color.fromRGBO(180, 180, 180, 1);
const uniGreen = Color.fromRGBO(0, 160, 130, 1);
const uniYellow = Color.fromRGBO(190, 170, 60, 1);
const uniPink = Color.fromRGBO(163, 83, 148, 1);
