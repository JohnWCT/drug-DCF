
from tools.round19_novelty_classifier import Round19NoveltyClassifier


def classifier():
    return Round19NoveltyClassifier(
        ["drug-a", "drug-b"],
        ["c1ccccc1", "CCO"],
        ["LUAD", "BRCA"],
    )


def test_novelty_priority_and_unknown_metadata():
    c = classifier()
    assert c.classify("drug-x", "c1ccccc1", "LUAD").novelty_class == "unseen_drug"
    assert c.classify("drug-a", "C1CCCCC1", "LUAD").novelty_class == "unseen_scaffold"
    assert c.classify("drug-a", "c1ccccc1", "PAAD").novelty_class == "unseen_cancer_type"
    assert c.classify("drug-a", "c1ccccc1", "LUAD").novelty_class == "source_like"
    unknown = c.classify(None, "c1ccccc1", "LUAD")
    assert unknown.novelty_class == "metadata_unknown"
    assert unknown.confidence == "low"
