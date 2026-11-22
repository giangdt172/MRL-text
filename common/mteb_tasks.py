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
        "FiQA2018",  # Finance QA
        "SCIDOCS",
        "ArguAna",
        "CQADupstackGamingRetrieval",
        "CQADupstackUnixRetrieval",
    ),
    "Classification": (
        "Banking77Classification",
        "TweetSentimentExtractionClassification",
        "ToxicConversationsClassification",
        "ImdbClassification",
        "MassiveIntentClassification",
    ),
    "Clustering": (
        "ArXivHierarchicalClusteringS2S",
        "MedrxivClusteringS2S.v2",
        "TwentyNewsgroupsClustering.v2",
        "BiorxivClusteringP2P.v2",
        "MedrxivClusteringP2P.v2",
    ),
}

DEFAULT_EVAL_TASKS = tuple(name for group in EVAL_TASK_GROUPS.values() for name in group)
