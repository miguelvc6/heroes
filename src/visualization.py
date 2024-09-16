import json
import logging
import os
import random
from typing import List, Dict, Tuple

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from umap import UMAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(characters_file: str, embeddings_file: str, max_items: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load and merge data from character and embedding files.

    Args:
        characters_file (str): Path to the characters JSON file.
        embeddings_file (str): Path to the embeddings JSON file.
        max_items (int, optional): Maximum number of items to return. Defaults to None.

    Returns:
        Tuple[List[Dict], List[Dict]]: Merged data and heroes data.
    """
    logging.info("Loading data from files...")
    with open(characters_file, "r", encoding="utf-8") as f:
        heroes = [json.loads(line) for line in f]

    with open(embeddings_file, "r", encoding="utf-8") as f:
        embedding_list = json.load(f)
    output_file = "data/merged_data.json"

    # Check if merged data already exists
    if os.path.exists(output_file):
        logging.info(f"Loading merged data from {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            merged_data = json.load(f)
    else:
        logging.info("Merging data...")
        # Merge hero data with embeddings
        merged_data = [
            {**e, **h} for e in embedding_list for h in heroes if e["title"] == h["title"] and e["type"] == h["type"]
        ]
        logging.info("Exporting merged data...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Merged data exported to {output_file}")

    logging.info(f"Data loaded. Merged data count: {len(merged_data)}")
    random.shuffle(merged_data)
    merged_data = merged_data[:max_items]
    return merged_data, heroes


def extract_data(merged_data: List[Dict]) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Extract relevant data from merged dataset.

    Args:
        merged_data (List[Dict]): List of merged data dictionaries.

    Returns:
        Tuple[np.ndarray, List[str], List[str], List[str]]: Embeddings, titles, types, and URLs.
    """
    logging.info("Extracting data from merged dataset...")
    embeddings = np.array([item["embedding"] for item in merged_data])
    titles = [item["title"] for item in merged_data]
    types = [item["type"] for item in merged_data]
    urls = [item["url"] for item in merged_data]
    logging.info(f"Data extracted. Number of items: {len(titles)}")
    return embeddings, titles, types, urls


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings using StandardScaler.

    Args:
        embeddings (np.ndarray): Input embeddings.

    Returns:
        np.ndarray: Normalized embeddings.
    """
    logging.info("Normalizing embeddings...")
    scaler = StandardScaler()
    normalized = scaler.fit_transform(embeddings)
    logging.info("Embeddings normalized")
    return normalized


def reduce_dimensionality(embeddings_normalized: np.ndarray) -> np.ndarray:
    """
    Reduce dimensionality of normalized embeddings using UMAP.

    Args:
        embeddings_normalized (np.ndarray): Normalized embeddings.

    Returns:
        np.ndarray: Reduced 3D embeddings.
    """
    logging.info("Reducing dimensionality with UMAP...")
    # UMAP parameters can be adjusted for different visualizations
    umap_3d = UMAP(n_components=3, n_neighbors=15, min_dist=0.5, spread=1.5, n_jobs=-1)
    reduced = umap_3d.fit_transform(embeddings_normalized)
    logging.info("Dimensionality reduction completed")
    return reduced


def create_color_map(types: List[str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """
    Create a color map for different types of data points.

    Args:
        types (List[str]): List of data point types.

    Returns:
        Tuple[List[str], Dict[str, str], List[str]]: Colors, color map, and unique types.
    """
    logging.info("Creating color map...")
    unique_types = list(set(types))
    colorscale = pc.sequential.Viridis
    # Map each unique type to a color in the Viridis colorscale
    color_map = {
        t: pc.sample_colorscale(colorscale, i / (len(unique_types) - 1))[0] for i, t in enumerate(unique_types)
    }
    colors = [color_map[t] for t in types]
    logging.info(f"Color map created. Unique types: {len(unique_types)}")
    return colors, color_map, unique_types


def create_3d_scatter_plot(
    proj_3d: np.ndarray, colors: List[str], titles: List[str], types: List[str], urls: List[str]
) -> go.Figure:
    """
    Create a 3D scatter plot using Plotly.

    Args:
        proj_3d (np.ndarray): 3D projected data points.
        colors (List[str]): List of colors for each data point.
        titles (List[str]): List of titles for each data point.
        types (List[str]): List of types for each data point.
        urls (List[str]): List of URLs for each data point.

    Returns:
        go.Figure: Plotly Figure object containing the 3D scatter plot.
    """
    logging.info("Creating 3D scatter plot...")
    fig = go.Figure(
        data=go.Scatter3d(
            x=proj_3d[:, 0],
            y=proj_3d[:, 1],
            z=proj_3d[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors, opacity=0.8, line=dict(width=0.5, color="DarkSlateGrey")),
            text=[f"Title: {title}<br>Type: {type}<br>URL: {url}" for title, type, url in zip(titles, types, urls)],
            hoverinfo="text",
        )
    )
    logging.info("3D scatter plot created")
    return fig


def update_layout(fig: go.Figure, unique_types: List[str], color_map: Dict[str, str], titles: List[str]) -> None:
    """
    Update the layout of the Plotly Figure.

    Args:
        fig (go.Figure): Plotly Figure object to update.
        unique_types (List[str]): List of unique data point types.
        color_map (Dict[str, str]): Mapping of types to colors.
        titles (List[str]): List of titles for each data point.
    """
    logging.info("Updating plot layout...")
    fig.update_layout(
        title="3D Hero Embeddings Visualization",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            aspectmode="cube",
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1, y=1, z=1)),
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        font=dict(family="Arial", size=12, color="white"),
        paper_bgcolor="rgba(0,0,0,0.95)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        legend=dict(
            itemsizing="constant",
            bgcolor="rgba(50,50,50,0.7)",
            bordercolor="rgba(200,200,200,0.5)",
            borderwidth=1,
        ),
    )

    # Add legend items for each unique type
    for t in unique_types:
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=6, color=color_map[t]),
                name=t,
            )
        )
    logging.info("Layout updated")


def inject_search_functionality(html_content: str) -> str:
    """
    Inject search functionality into the HTML content.

    Args:
        html_content (str): Original HTML content.

    Returns:
        str: HTML content with search functionality injected.
    """
    # JavaScript for search functionality
    search_script = """
    <script>
    var originalColors = [];
    function searchPoints() {
        console.log("searchPoints function called");
        var myDiv = document.getElementById('myDiv');
        if (!myDiv || !myDiv.data || !myDiv.data[0]) {
            console.error("Plotly graph not found");
            return;
        }
        
        var searchTerm = document.getElementById('search-box').value.toLowerCase();
        console.log("Search term:", searchTerm);
        var data = myDiv.data[0];
        var update = {
            'marker.color': [],
            'marker.size': []
        };
        for (var i = 0; i < data.text.length; i++) {
            if (searchTerm !== '' && data.text[i].toLowerCase().includes(searchTerm)) {
                update['marker.color'].push('red');
                update['marker.size'].push(6);
            } else {
                update['marker.color'].push(originalColors[i]);
                update['marker.size'].push(2);
            }
        }
        console.log("Update object:", update);
        Plotly.restyle('myDiv', {
            'marker.color': [update['marker.color']],
            'marker.size': [update['marker.size']]
        }, [0]);
        console.log("Plotly restyle called");
    }

    function initializeSearch() {
        console.log("initializeSearch function called");
        var myDiv = document.getElementById('myDiv');
        if (myDiv && myDiv.data && myDiv.data[0]) {
            var data = myDiv.data[0];
            // Store original colors
            originalColors = data.marker.color.slice();
            document.getElementById('search-box').addEventListener('input', searchPoints);
            console.log("Search box event listener added");
            return true;
        }
        console.warn("Plotly graph not ready yet");
        return false;
    }

    function waitForPlotly() {
        if (!initializeSearch()) {
            setTimeout(waitForPlotly, 100);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', waitForPlotly);
    } else {
        waitForPlotly();
    }
    </script>
    """

    # CSS styles for search box
    search_styles = """
    <style>
    #search-container {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 5px;
        border-radius: 5px;
        display: flex;
        align-items: center;
    }

    #search-container label {
        margin-right: 5px;
        font-weight: bold;
    }

    #search-container input[type="text"] {
        padding: 5px;
        border: 1px solid #ccc;
        border-radius: 3px;
    }
    </style>
    """

    # HTML for search input
    search_input = """
    <div id="search-container">
        <label for="search-box">Search:</label>
        <input type="text" id="search-box" placeholder="Search heroes...">
    </div>
    """

    # Inject the CSS styles into the head
    html_content = html_content.replace("</head>", f"{search_styles}{search_script}</head>")
    # Inject the search input into the body
    html_content = html_content.replace("<body>", f"<body>{search_input}")
    return html_content


def main():
    """
    Main function to orchestrate the data processing and visualization pipeline.
    """
    logging.info("Starting main function...")
    merged_data, heroes = load_data("data/characters.jsonl", "data/hero_embeddings.json", max_items=None)
    print(f"Number of merged items: {len(merged_data)}")

    friends_count = sum(1 for item in heroes if item.get("friends"))
    print(f"Number of instances with non-empty 'friends' list: {friends_count}")

    embeddings, titles, types, urls = extract_data(merged_data)
    embeddings_normalized = normalize_embeddings(embeddings)
    proj_3d = reduce_dimensionality(embeddings_normalized)
    colors, color_map, unique_types = create_color_map(types)

    fig = create_3d_scatter_plot(proj_3d, colors, titles, types, urls)
    update_layout(fig, unique_types, color_map, titles)

    logging.info("Writing HTML file...")
    html_file = "3d_hero_embeddings_visualization_js.html"
    fig.write_html(html_file, include_plotlyjs="cdn", div_id="myDiv")

    # Inject search functionality
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    html_content = inject_search_functionality(html_content)

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logging.info("HTML file written with search functionality. Process complete.")


if __name__ == "__main__":
    main()
