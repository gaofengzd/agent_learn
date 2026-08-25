from importlib import import_module


def test_architecture_boundary_packages_are_importable() -> None:
    for package in (
        "paper_read_agent.application",
        "paper_read_agent.document_pipeline",
        "paper_read_agent.retrieval",
        "paper_read_agent.llm",
        "paper_read_agent.persistence",
        "paper_read_agent.ui",
    ):
        assert import_module(package) is not None
