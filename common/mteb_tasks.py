"""The selected 21-task English MTEB preset, shared by every method."""

EVAL_TASK_GROUPS = {
    "STS": (
        "STSBenchmark",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
    ),
    "Retrieval": (
        "QuoraRetrieval",
        "ArguAna",
        "TRECCOVID",
        "FEVER",
    ),
    "Classification": (
        "Banking77Classification",
        "ToxicConversationsClassification",
        "EmotionClassification",
        "MassiveIntentClassification",
    ),
    "Clustering": (
        "StackExchangeClustering",
        "ArxivClusteringS2S",
        "TwentyNewsgroupsClustering",
        "RedditClustering",
        "MedrxivClusteringS2S",
        "BiorxivClusteringS2S",
    ),
}

DEFAULT_EVAL_TASKS = tuple(name for group in EVAL_TASK_GROUPS.values() for name in group)
