# ANP-based Bike Network Evaluation

This repository provides a framework for evaluating bike networks by fusing Knowledge Graphs (KGs) and Multi Criteria Decision Analysis (MCDA).

**Key objectives:**
- Identify key evaluation metrics and criteria based on knowledge base of bike network evaluation studies.
- Structure bike network evaluation as a decision modelling task and address interactions between commonly used metrics and criteria.
- Derive a bikeability index for Zurich's road network.

## Features

1. **Leveraging Bike Network Evaluations' KG**
   Parse over 270 bike network metrics and over 40 qualitative criteria from KG and use it to determine preferences.
2. **Analytic Network Process**  
   Use Analytic Network Process (ANP), an MCDA technique, to derive priority weights for criteria, metrics and rank edges based on matrix operations.
3. **Bikeability Index**  
   Develop a Bikeability Index based on scholarly consensus and perform sensitivity analysis on how KG structure affects index outcomes.

## Installation

1. **Setup Environment**  
   Ensure you have `Python 3.11` or higher and `conda` is installed.
   Clone the repository and set up the Python environment.

   ```
   git clone <repository-url>
   cd <repository-directory>
   conda env create -f environment.yml
   conda activate anp
    ```

   2. **Directory Structure**:
      ```
      ├── main.py                 # runs the entire workflow.
      ├── anp_utils.py            # Functions to construct required matrices for the NAP approach.
      ├── context_data_load.py    # Functions to load the relevant contextual information.
      ├── data_preprocessing.py   # Functions to preprocess relevant contextual data.
      ├── constants.py            # mappings for some of the data that are described in categorical terms.
      └── data_submission/        # Contains input or generated output data
         ├── input/               # Intermediate data necessary for full workflow reproducibility
           ├── metrics.nq         # N-quads to be stored in a graph database.
           ├── ...                # rest of the contextual data.
           ├── zurich_kreise/     # Dataset with Zurich's districts.
         ├── output/              # All generated data from this workflow is stored here
      ```
- data_submission/input/: Contains the essential input files (e.g., contextual data, road network, metrics n-quad file to load into KG.
- data_submission/output/: Stores generated data such as bikeability map and sensitivity analysis results relevant for the ANP-based bikeability evaluation paper.

## Data and Reproducibility:

   - Input data specific to this workflow can be [downloaded](https://doi.org/10.5281/zenodo.14839760) and should be placed in the `data_submission/input/` folder to match the directory structure above.
   - The input data was generated using methodologies described in: metrics [[1]](#references) and road network [[2,3]](#references) and as part of the _blinded for peer review_ project. Contextual data was downloaded from [Zurich City Data Registry](https://data.stadt-zuerich.ch/) together with meta-information. 
   - The full workflow has been tested on Windows operating system, Lenovo ThinkPad X1 (12th Gen Intel(R) Core(TM) i7-1260P 2.10 GHz) and took up to 10 min.

## Knowledge Graph Setup:

**Install Blazegraph**:
- Ensure `Java 8+` is installed.
- [Download](https://github.com/blazegraph/database/releases/tag/BLAZEGRAPH_2_1_6_RC) `blazegraph.jar` and place it in the root directory.
- Start Blazegraph by navigating to the root directory and running the following in CLI:

  ```
  java -server -Xmx16g -jar blazegraph.jar
  ```
  
- Check that the **port** in the running local Blazegraph instance URL, e.g., `http://127.0.0.1:9999/blazegraph/` matches the port of the `kg` key in the `config.ini` file.
- You can now open the local KG URL in a browser. 
  - Under **NAMESPACES** tab create a new namespace called, e.g., `anp`, with Mode set to `quads`. Note that the namespace's name needs to match with the name in the `kg` key that appears between `/namespace` and `/sparql`. 
  - Under the **UPDATE** tab choose the `metrics.nq` file from the `input/` directory as Type `RDF data` and Format set to `N-Quads` and clicking `Update`. 

## ANP workflow:

Run the following with the ANP workflow that loads contextual data, enriches road network and constructs a limit matrix with final priority vectors and edge ranking. 
    
  ```
  python main.py
  ```

The priority vectors, bikeability maps and sensitivity analysis will be stored the `output/` directory.

## Tips and Troubleshooting:

- Ensure Blazegraph namespaces match those specified in the configuration. For the input files, do not change names.
- For namespace-specific issues, consult the Blazegraph wiki or ensure proper SPARQL endpoint connectivity.

## Acknowledgements:
_blinded for peer review_

## References:

[1]: blinded for review

[2]: blinded for review

[3]: blinded for review