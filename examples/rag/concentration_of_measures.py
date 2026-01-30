"""examples.rag.concentration_of_measures"""

from sv.rag.concentration_of_measures import ConcentrationOfMeasures


if __name__ == "__main__":
    ConcentrationOfMeasures(
        n=1000,
        dimensions=[2, 10, 50, 100, 500, 1000, 1536],
        reference=None,
        output_path="examples/rag/concentration_of_measures.png",
        overlay=False,
    ).run()
