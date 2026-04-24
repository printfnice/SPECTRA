"""
Knowledge Graph Metadata Extraction for HGATLink Enhancement
知识图谱元数据提取模块

This module extracts prior knowledge from LIANA resources to enhance
HGATLink's ligand-receptor interaction predictions.

Key features:
1. Extract confidence scores from multiple databases (consensus, CellPhoneDB, OmniPath, etc.)
2. Score LR pairs based on database frequency (high consensus = high confidence)
3. Cache scores for efficient reuse
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class KGMetadataExtractor:
    """
    Extract knowledge graph metadata from LIANA resources
    从LIANA资源提取知识图谱元数据

    Scoring strategy based on database consensus:
    - Present in 3+ databases → 1.0 (high confidence)
    - Present in 2 databases → 0.7 (medium confidence)
    - Present in 1 database → 0.5 (low confidence)
    - Not in any database → 0.3 (data-driven, low prior)
    """

    def __init__(self, resources: List[str] = None):
        """
        Initialize KG metadata extractor

        Args:
            resources: List of LIANA resource names to use
                      Default: ['consensus', 'CellPhoneDB', 'OmniPath', 'CellChatDB']
        """
        if resources is None:
            # Use multiple resources for better coverage
            resources = ['consensus', 'CellPhoneDB', 'OmniPath', 'CellChatDB']

        self.resources = resources
        self.kg_scores_cache = {}
        self.resource_data = {}
        self.loaded = False

    def load_resources(self, verbose: bool = True):
        """
        Load LIANA resources from the liana package

        Args:
            verbose: Whether to print loading progress
        """
        try:
            import liana as li
        except ImportError:
            raise ImportError(
                "liana package is required for KG metadata extraction. "
                "Please install it with: pip install liana"
            )

        if verbose:
            print(f"Loading {len(self.resources)} LIANA resources for KG metadata...")

        for resource_name in self.resources:
            try:
                resource_df = li.resource.select_resource(resource_name)
                self.resource_data[resource_name] = resource_df

                if verbose:
                    print(f"  ✓ {resource_name}: {len(resource_df)} LR pairs")

            except Exception as e:
                if verbose:
                    print(f"  ✗ {resource_name}: Failed to load ({e})")
                continue

        if len(self.resource_data) == 0:
            raise RuntimeError("Failed to load any LIANA resources")

        self.loaded = True

        if verbose:
            print(f"✓ Loaded {len(self.resource_data)} resources successfully")

    def _get_lr_key(self, ligand: str, receptor: str) -> Tuple[str, str]:
        """
        Normalize LR pair to a consistent key format

        Args:
            ligand: Ligand gene name or complex
            receptor: Receptor gene name or complex

        Returns:
            Tuple of (ligand, receptor) as key
        """
        # Handle complex notation (e.g., "CCL2_CCL3")
        # Keep as-is for now, could add complex splitting logic later
        return (ligand, receptor)

    def extract_scores(self,
                       lr_pairs: pd.DataFrame = None,
                       ligands: List[str] = None,
                       receptors: List[str] = None,
                       verbose: bool = True) -> Dict[Tuple[str, str], float]:
        """
        Extract KG confidence scores for LR pairs

        Args:
            lr_pairs: DataFrame with 'ligand' and 'receptor' columns (from LIANA resource)
            ligands: List of ligand genes (alternative input)
            receptors: List of receptor genes (alternative input)
            verbose: Whether to print extraction progress

        Returns:
            Dictionary mapping (ligand, receptor) tuples to confidence scores (0-1)

        Scoring logic:
            - 3+ databases: 1.0 (highest confidence)
            - 2 databases: 0.7 (medium confidence)
            - 1 database: 0.5 (low confidence)
            - 0 databases: 0.3 (pure data-driven, minimal prior)
        """
        if not self.loaded:
            self.load_resources(verbose=verbose)

        # Prepare LR pairs to score
        if lr_pairs is not None:
            # Use DataFrame input
            lr_list = list(zip(lr_pairs['ligand'], lr_pairs['receptor']))
        elif ligands is not None and receptors is not None:
            # Use list inputs
            lr_list = [(lig, rec) for lig in ligands for rec in receptors]
        else:
            raise ValueError("Must provide either lr_pairs DataFrame or ligands+receptors lists")

        if verbose:
            print(f"\nExtracting KG scores for {len(lr_list)} LR pairs...")

        kg_scores = {}

        # Count occurrences of each LR pair across resources
        lr_occurrence_count = {}

        for resource_name, resource_df in self.resource_data.items():
            # Create set of LR pairs in this resource for fast lookup
            resource_lr_set = set(
                zip(resource_df['ligand'], resource_df['receptor'])
            )

            # Count occurrences
            for lr in lr_list:
                lr_key = self._get_lr_key(lr[0], lr[1])

                if lr_key in resource_lr_set:
                    lr_occurrence_count[lr_key] = lr_occurrence_count.get(lr_key, 0) + 1

        # Convert occurrence counts to confidence scores
        for lr in lr_list:
            lr_key = self._get_lr_key(lr[0], lr[1])
            count = lr_occurrence_count.get(lr_key, 0)

            # Apply scoring strategy
            if count >= 3:
                score = 1.0  # High confidence (multiple databases)
            elif count == 2:
                score = 0.7  # Medium confidence
            elif count == 1:
                score = 0.5  # Low confidence (single database)
            else:
                score = 0.3  # No database support (data-driven fallback)

            kg_scores[lr_key] = score

        # Cache the results
        self.kg_scores_cache.update(kg_scores)

        if verbose:
            print(f"✓ KG scores extracted for {len(kg_scores)} LR pairs")

            # Statistics
            score_dist = pd.Series(list(kg_scores.values())).value_counts().sort_index(ascending=False)
            print(f"\nScore distribution:")
            for score, count in score_dist.items():
                pct = 100 * count / len(kg_scores)
                if score == 1.0:
                    label = "High confidence (3+ DBs)"
                elif score == 0.7:
                    label = "Medium confidence (2 DBs)"
                elif score == 0.5:
                    label = "Low confidence (1 DB)"
                else:
                    label = "No DB support (data-driven)"
                print(f"  {score:.1f} ({label}): {count} ({pct:.1f}%)")

        return kg_scores

    def get_score(self, ligand: str, receptor: str, default: float = 0.5) -> float:
        """
        Get KG confidence score for a single LR pair

        Args:
            ligand: Ligand gene name
            receptor: Receptor gene name
            default: Default score if not found in cache

        Returns:
            Confidence score (0-1)
        """
        lr_key = self._get_lr_key(ligand, receptor)
        return self.kg_scores_cache.get(lr_key, default)

    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded KG metadata

        Returns:
            Dictionary with statistics
        """
        if not self.loaded:
            return {
                'loaded': False,
                'n_resources': 0,
                'n_lr_pairs': 0
            }

        total_lr_pairs = sum(len(df) for df in self.resource_data.values())

        # Get unique LR pairs across all resources
        all_lr_pairs = set()
        for resource_df in self.resource_data.values():
            lr_pairs = set(zip(resource_df['ligand'], resource_df['receptor']))
            all_lr_pairs.update(lr_pairs)

        return {
            'loaded': True,
            'n_resources': len(self.resource_data),
            'n_lr_pairs_total': total_lr_pairs,
            'n_lr_pairs_unique': len(all_lr_pairs),
            'resources': list(self.resource_data.keys()),
            'n_cached_scores': len(self.kg_scores_cache)
        }


def test_kg_extractor():
    """
    Simple test function to verify KG metadata extraction
    """
    print("=" * 60)
    print("Testing KG Metadata Extractor")
    print("=" * 60)

    # Initialize extractor
    extractor = KGMetadataExtractor()

    # Load resources
    extractor.load_resources(verbose=True)

    # Print statistics
    stats = extractor.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test with some known LR pairs
    test_pairs = pd.DataFrame({
        'ligand': ['CCL2', 'VEGFA', 'TGFB1', 'UNKNOWN_GENE'],
        'receptor': ['CCR2', 'FLT1', 'TGFBR1', 'UNKNOWN_RECEPTOR']
    })

    print("\nTesting with sample LR pairs:")
    scores = extractor.extract_scores(test_pairs, verbose=True)

    print("\nIndividual scores:")
    for (lig, rec), score in scores.items():
        print(f"  {lig} -> {rec}: {score:.2f}")

    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_kg_extractor()
