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
        "SciFact",
        "ArguAna",
        "NFCorpus",    ),
    "Classification": (
        "Banking77Classification",
        "ToxicConversationsClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
    ),
    "Clustering": (
        "ArXivHierarchicalClusteringS2S",
        "MedrxivClusteringS2S",
        "TwentyNewsgroupsClustering",
        "BiorxivClusteringP2P",
        "MedrxivClusteringP2P",
    ),
}

DEFAULT_EVAL_TASKS = tuple(name for group in EVAL_TASK_GROUPS.values() for name in group)
