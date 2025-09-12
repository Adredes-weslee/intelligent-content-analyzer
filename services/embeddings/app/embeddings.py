"""Placeholder embedding functions.

In a full implementation this module would wrap a machine learning
library or a remote API to compute sentence embeddings. It could also
support different embedding models depending on configuration. For this
mini project we provide a single stub function that generates random
vectors. Keeping the code in a separate module allows the API router
to remain clean and makes it easy to replace with a real model later.
"""

from __future__ import annotations

from typing import List
import numpy as np


def embed_texts(texts: List[str], dim: int = 128) -> List[List[float]]:
    """Return a list of random embedding vectors.

    Args:
        texts: A list of strings to embed.
        dim: Dimensionality of the returned vectors.

    Returns:
        A list of lists of floats representing the embeddings.
    """
    return [np.random.rand(dim).tolist() for _ in texts]