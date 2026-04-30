# Molecular Networking

!!! danger "Experimental Feature"
    GraphML molecular networking is currently classified as an **Experimental** feature in MassFlow. It is not part of the stable v1.0 pipeline contract. The generated outputs, node properties, and edge attributes are provided as-is and are subject to breaking changes in future releases.

MassFlow includes experimental support for exporting your annotation results as a molecular network. This allows you to visualize the relationships between your experimental query spectra and the reference library in tools like [Cytoscape](https://cytoscape.org/) or [Gephi](https://gephi.org/).

## How It Works

When networking is enabled, MassFlow builds a `NetworkX` graph using two distinct types of relationships:

1.  **Query-to-Reference Edges:** These edges are derived directly from the successful annotation hits computed during your main workflow. They connect your experimental spectra to their matched library standards, using the similarity score as the edge weight.
2.  **Query-to-Query Edges:** To cluster unknown spectra, MassFlow performs an "all-vs-all" similarity calculation between your processed experimental query spectra. It uses the same configured similarity algorithm (e.g., `cosine`) to compute symmetric scores and draws edges between unknown queries that are structurally similar.

The resulting graph contains both "query" nodes and "reference" nodes, allowing annotation hits and *de novo* query clusters to appear in the same exported view.

---

## Enabling Molecular Networking

To generate a network, you must toggle the feature flag in the `workflow` section of your YAML configuration file:

```yaml
workflow:
  perform_networking: true
```

When you run `massflow annotate --config massflow_config.yaml` with this flag enabled, MassFlow will perform the standard annotation search first. After the CSV results are generated, it will compute the query-to-query matrix and export a GraphML file.

## Output Structure

The generated network is saved into your `project.output_directory` as:

*   `molecular_network.graphml`

### Node Attributes

Every node in the GraphML file contains the following properties to assist with downstream visualization styling:

*   `id`: The unique identifier (e.g., `query_0` or the reference `id`).
*   `node_type`: Either `"query"` or `"reference"`. You can use this attribute in Cytoscape to color your experimental data differently from the library standards.
*   `precursor_mz`: The parsed mass-to-charge ratio of the spectrum.
*   `name`: For reference nodes, this contains the `compound_name`.

### Edge Attributes

Every edge contains the following properties:

*   `weight`: The computed similarity score (e.g., the Cosine score).
*   `edge_type`: Either `"query_to_ref"` (an annotation hit) or `"query_to_query"` (a structural similarity cluster).

---

## Important Constraints & Limitations

Because this is an experimental feature, there are several known constraints:

1.  **Memory Consumption:** The query-to-query calculation requires building an $N \times N$ dense distance matrix in memory, where $N$ is the total number of experimental query spectra. If you process tens of thousands of queries, this step may exhaust your system's RAM.
2.  **Engine Compatibility:** The query-to-query matrix calculation requires the configured similarity engine to support symmetric scoring. If you are using a complex `ConsensusEngine` or `CascadeEngine`, MassFlow will attempt to fall back to the first underlying sub-engine (typically `cosine`) to build the network. If it cannot find a compatible scoring function, it will log a warning and abort the network generation.
3.  **Score Thresholds:** The query-to-query edges are strictly filtered using the same `similarity.min_score` threshold defined for the main annotation run. Edges below this score are dropped to keep the graph sparse.
