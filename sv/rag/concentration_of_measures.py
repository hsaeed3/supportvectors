"""sv.rag.concentration_of_measures

Provides a set of functions/utilities that help visualize the concept of
'concentration of measures', in high-dimensional spaces.
"""

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional

import numpy as np

__all__ = (
    "ConcentrationOfMeasures",
    "calculate_concentration_across_dimensions",
    "plot_concentration_of_measures",
)


@dataclass
class ConcentrationOfMeasures:
    """
    Unneccassary helper class that provides a structured interface for
    configuring the concentration of measures example.
    """

    n: int = 1000
    """The number of random vectors to be generated for each dimensionality that
    will be tested."""

    dimensions: List[int] = field(default_factory=lambda: [2, 10, 50, 100, 500, 1000, 1536])
    """The list of dimensionalities to be tested. The cosine similarities for each
    of these will be calculated and plotted, showcasing the difference as
    dimensionality increases."""

    reference: Optional[int | np.ndarray] = None
    """The reference vector to be used for calculating the cosine similarities.
    If None, a random vector from the set of vectors will be used."""

    output_path: str = "concentration_of_measures.png"
    """The path to save the image of the plot to. Please ensure this ends in
    .png"""

    overlay: bool = False
    """Whether or not to plot a single overlaid histogram, or multiple
    separate histograms."""

    def run(self) -> str:
        """
        Run the example.

        Returns
        -------
        str
            The path to the saved plot.
        """
        try:
            results = calculate_concentration_across_dimensions(
                n=self.n, dimensions=self.dimensions, reference=self.reference
            )
            plot_concentration_of_measures(
                results=results, output_path=self.output_path, overlay=self.overlay
            )
        except Exception as e:
            raise ValueError(
                f"An error occurred while running the concentration of measures example: {e}"
            ) from e

        return self.output_path


def calculate_cosine_similarities(
    vectors: np.ndarray, reference: Optional[int | np.ndarray] = None
) -> np.ndarray:
    """
    Calculates cosine similarities for a given or random reference
    vector across a set of vectors, using numpy.

    Parameters
    ----------
    vectors : np.ndarray
        A numpy array of shape (n, dimensions) containing the vectors.
    reference : Optional[int | np.ndarray], optional
        Either the index of the reference vector within `vectors`, or an
        explicit reference vector (1D array). If None, a random vector
        from the set is selected. Defaults to None.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n,) containing the cosine similarities.
    """
    # my brother why are we only providing 1 vector?
    if len(vectors) <= 1:
        raise ValueError(
            f"Expected at least 2 vectors to calculate cosine similarities, "
            f"got {len(vectors)}"
        )

    if reference is None:
        # pick out a random reference vector from the set of vectors
        # if one hasnt been provided
        reference_idx: int = random.randint(a=0, b=vectors.shape[0] - 1)
        reference_vector = vectors[reference_idx]

    elif isinstance(reference, (int, np.integer)):
        reference_vector = vectors[reference]

    else:
        reference_vector = np.asarray(reference)

    # normalize vectors and reference vector
    normalized_vectors = np.linalg.norm(x=vectors, axis=1)
    normalized_reference = np.linalg.norm(x=reference_vector)

    # ah math!
    return np.dot(a=vectors, b=reference_vector) / (
        normalized_vectors * normalized_reference
    )


def calculate_concentration_across_dimensions(
    n: int = 1000,
    dimensions: Optional[List[int]] = None,
    reference: Optional[int | np.ndarray] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Calculates the concentration of measures by computing cosing similarity
    between a set of random vectors and a reference vector, across different
    dimensionalities.

    This is to show that as dimensionality increases, the you can
    'see' how *most vectors will sit closer and closer to the
    equatorial plane.

    Parameters
    ----------
    n : int, optional
        Number of random vectors to generate for each dimensionality.
        Defaults to 1000.
    dimensions : list[int], optional
        List of dimensionalities to test. Defaults to [2, 10, 50, 100, 500, 1000, 1536].
    reference : Optional[int], optional
        Index of the reference vector to use consistently across all
        dimensionalities. If None, a random one is picked. Defaults to None.

    Returns
    -------
    Dict[int, Dict[str, float]]
        A dictionary containing the concentration of measures for each dimensionality.
    """
    # could provide seed as a param, but not necessary this is a toy
    # example
    seed: int = 42

    if n is None or n <= 1:
        raise ValueError(
            "Expected a value of 2 or greater for the number of vectors to generate, "
            f"received n={n}."
        )

    if dimensions is None:
        raise ValueError(f"Please provide a list of dimensionalities to test.")

    if reference is None:
        # make one up
        reference: int = random.randint(a=0, b=n - 1)

    results: Dict[int, Dict[str, float]] = {}

    for d in dimensions:
        # generate random vectors
        vectors = np.random.randn(n, d)
        cosine_similarities = calculate_cosine_similarities(
            vectors=vectors, reference=reference
        )

        # exclude similarity to reference vector
        mask = np.ones(len(cosine_similarities), dtype=bool)
        mask[reference] = False
        filtered_similarities = cosine_similarities[mask]

        results[d] = {
            "similarities": filtered_similarities,
            "mean": float(np.mean(filtered_similarities)),
            "std": float(np.std(filtered_similarities)),
            "theoretical_std": 1.0 / np.sqrt(d),
        }

    return results


def plot_concentration_of_measures(
    results: Dict[int, Dict[str, float]],
    output_path: str = "concentration_of_measures.png",
    overlay: bool = False,
) -> str:
    """
    Plots the concentration of measures concept within an image of
    multiple histograms, or within a single overlaid histogram,
    based on the `overlay` parameter.

    Parameters
    ----------
    results : Dict[int, Dict[str, float]]
        Output from `calculate_concentration_across_dimensions()`.
    output_path : str, optional
        Path to save the plot. Defaults to "concentration_of_measures.png".
    overlay : bool, optional
        Whether to plot a single overlaid histogram. Defaults to False.

    Returns
    -------
    str
        The path to the saved plot.
    """
    import matplotlib.pyplot as plt

    dims = sorted(results.keys())
    n_dims = len(dims)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_dims))  # type: ignore

    if overlay:
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, d in enumerate(dims):
            sims = results[d]["similarities"]
            ax.hist(
                sims,
                bins=50,
                alpha=0.5,
                label=f"d={d}",
                color=colors[i],
                density=True,
                range=(-1, 1),
            )

        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Equator")
        ax.set_xlabel("Cosine Similarity to Reference Vector", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Concentration of Measures", fontsize=13)
        ax.set_xlim(-1, 1)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    else:
        cols = 3
        rows = (n_dims + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.flatten()

        for i, d in enumerate(dims):
            ax = axes[i]
            sims = results[d]["similarities"]
            ax.hist(
                sims, bins=50, color=colors[i], density=True, range=(-1, 1), alpha=0.7
            )
            ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
            ax.set_title(f"d={d} (σ={results[d]['std']:.3f})", fontsize=11)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)

        # hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Concentration of Measures Across Dimensions", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
