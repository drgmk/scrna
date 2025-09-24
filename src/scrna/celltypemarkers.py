import json
import os
import pandas as pd
from typing import Dict, List, Union, Optional


class CellTypeMarkers:
    """A class to handle cell type marker genes with support for short names and case conversion.

    todo: want to add specificity so it works better with decoupler
    """

    def __init__(self, organism: str, data: Optional[Dict] = None):
        """Initialize with marker gene data.

        Parameters
        ----------
        organism : str
            Target organism ('human' or 'mouse'). This determines gene name case:
            - 'human': uppercase genes (e.g., 'FOXP3')
            - 'mouse': title case genes (e.g., 'Foxp3')
        data : dict, optional
            Dictionary containing cell type marker data. If None, loads from package.
        """
        self.organism = organism

        # Set case based on organism
        if organism.lower() == "human":
            self.case = "upper"
        elif organism.lower() == "mouse":
            self.case = "title"
        else:
            raise ValueError(
                f"Unknown organism '{organism}'. Valid options are 'human' or 'mouse'."
            )

        if data is None:
            self.data = self._load_default_markers()
        else:
            self.data = data

        # Apply case conversion to all gene data at initialization
        self._apply_case_conversion()

    def _load_default_markers(self) -> Dict:
        """Load default marker genes from the package JSON file."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        marker_genes_path = os.path.join(module_dir, "data/marker_genes.json")

        try:
            with open(marker_genes_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Marker genes file not found at {marker_genes_path}")
            return {}

    def _apply_case_conversion(self):
        """Apply case conversion to all gene lists in the data."""
        if self.case is None:
            return

        for cell_type, cell_data in self.data.items():
            # Convert primary genes
            if "genes" in cell_data:
                if self.case.lower() == "upper":
                    cell_data["genes"] = [gene.upper() for gene in cell_data["genes"]]
                elif self.case.lower() == "title":
                    cell_data["genes"] = [
                        gene.capitalize() for gene in cell_data["genes"]
                    ]
                else:
                    print(
                        f"Warning: Unknown case option '{self.case}'. Valid options are 'upper', 'title', or None."
                    )

            # Convert secondary genes
            if "genes_secondary" in cell_data:
                if self.case.lower() == "upper":
                    cell_data["genes_secondary"] = [
                        gene.upper() for gene in cell_data["genes_secondary"]
                    ]
                elif self.case.lower() == "title":
                    cell_data["genes_secondary"] = [
                        gene.capitalize() for gene in cell_data["genes_secondary"]
                    ]

    def get_markers(self, cell_type: str, include_secondary: bool = False) -> List[str]:
        """Get marker genes for a cell type.

        Parameters
        ----------
        cell_type : str
            Cell type name.
        include_secondary : bool, optional
            If True, include secondary genes in addition to primary genes. Default is False.

        Returns
        -------
        list
            List of marker gene names (case already applied at initialization).
        """
        if cell_type not in self.data:
            available = list(self.data.keys())
            raise ValueError(
                f"Cell type '{cell_type}' not found. Available: {available}"
            )

        markers = self.data[cell_type]["genes"].copy()

        # Add secondary genes if requested
        if include_secondary:
            secondary_genes = self.data[cell_type].get("genes_secondary", [])
            markers.extend(secondary_genes)

        return markers

    def get_short_name(self, cell_type: str) -> str:
        """Get the short name for a cell type.

        Parameters
        ----------
        cell_type : str
            Cell type name.

        Returns
        -------
        str
            Short name for the cell type.
        """
        if cell_type not in self.data:
            available = list(self.data.keys())
            raise ValueError(
                f"Cell type '{cell_type}' not found. Available: {available}"
            )

        return self.data[cell_type]["short_name"]

    def get_secondary_markers(self, cell_type: str) -> List[str]:
        """Get secondary marker genes for a cell type.

        Parameters
        ----------
        cell_type : str
            Cell type name.

        Returns
        -------
        list
            List of secondary marker gene names (case already applied at initialization).
        """
        if cell_type not in self.data:
            available = list(self.data.keys())
            raise ValueError(
                f"Cell type '{cell_type}' not found. Available: {available}"
            )

        return self.data[cell_type].get("genes_secondary", []).copy()

    def filter_genes(
        self,
        gene_names: List[str],
        cell_types: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> None:
        """Filter marker genes in-place to only include those present in gene list.

        This method modifies the object by removing genes from both primary and secondary
        gene lists that are not found in the provided gene names.

        Parameters
        ----------
        gene_names : list
            List of gene names to filter against (e.g., adata.var_names).
        cell_types : list, optional
            List of cell types to filter. If None, filters all.
        verbose : bool, optional
            If True, print information about missing genes. Default is True.
        """
        if cell_types is None:
            cell_types = list(self.data.keys())

        gene_set = set(gene_names)

        for cell_type in cell_types:
            if cell_type not in self.data:
                continue

            # Filter primary genes
            original_genes = self.data[cell_type]["genes"].copy()
            filtered_genes = [gene for gene in original_genes if gene in gene_set]
            missing_primary = [gene for gene in original_genes if gene not in gene_set]
            self.data[cell_type]["genes"] = filtered_genes

            # Filter secondary genes
            original_secondary = self.data[cell_type].get("genes_secondary", []).copy()
            filtered_secondary = [
                gene for gene in original_secondary if gene in gene_set
            ]
            missing_secondary = [
                gene for gene in original_secondary if gene not in gene_set
            ]
            self.data[cell_type]["genes_secondary"] = filtered_secondary

            if verbose and (missing_primary or missing_secondary):
                missing_info = []
                if missing_primary:
                    missing_info.append(f"primary: {missing_primary}")
                if missing_secondary:
                    missing_info.append(f"secondary: {missing_secondary}")
                print(f"Missing genes in {cell_type} - {', '.join(missing_info)}")

    def to_dict(self, include_secondary: bool = False) -> Dict[str, List[str]]:
        """Convert to dictionary format (backward compatibility).

        Parameters
        ----------
        include_secondary : bool, optional
            If True, include secondary genes in addition to primary genes. Default is False.

        Returns
        -------
        dict
            Dictionary with cell type names as keys and gene lists as values.
        """
        result = {}
        for cell_type in self.data.keys():
            result[cell_type] = self.get_markers(
                cell_type, include_secondary=include_secondary
            )
        return result

    def to_pandas(self, include_secondary: bool = False):
        """Convert to pandas DataFrame in long format.

        Parameters
        ----------
        include_secondary : bool, optional
            If True, include secondary genes in addition to primary genes. Default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'cell_type' and 'gene' columns in long format.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_pandas method. Install with: pip install pandas"
            )

        rows = []
        for cell_type in self.data.keys():
            markers = self.get_markers(cell_type, include_secondary=include_secondary)
            for gene in markers:
                rows.append({"cell_type": cell_type, "gene": gene})

        return pd.DataFrame(rows)

    def __getitem__(self, key: str) -> List[str]:
        """Allow dictionary-like access."""
        return self.get_markers(key)

    def __contains__(self, key: str) -> bool:
        """Check if cell type exists."""
        return key in self.data

    def keys(self, min=1, include_secondary: bool = False):
        """Get all cell type names, checking if genes/secondary_genes lists are empty."""
        for k in self.data.keys():
            if len(self.get_markers(k, include_secondary=include_secondary)) >= min:
                yield k

    def items(self):
        """Iterate over cell type, marker pairs."""
        for cell_type in self.keys():
            yield cell_type, self.get_markers(cell_type)
