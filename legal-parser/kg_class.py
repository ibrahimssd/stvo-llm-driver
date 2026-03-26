import json
import networkx as nx
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from functools import partial   


class KnowledgeGraphProcessor:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    
    def draw_graph(self, graph, start_node=None, title='Graph Visualization', filename='./plots/graph.png', figsize=(12, 12)):
        """
        Draws a graph or a connected component from a specified start node with colorized nodes based on their ID types.
        
        :param graph: NetworkX graph object to visualize.
        :param start_node: Starting node ID to focus on the connected component. If None, draws the entire graph.
        :param title: Title of the graph to display.
        :param filename: Filename to save the graph image.
        :param figsize: Size of the figure in inches.
        """
        if start_node and start_node in graph:
            # Use BFS to find all nodes reachable from the start node
            reachable = nx.single_source_shortest_path_length(graph, start_node)
            subgraph = graph.subgraph(reachable)
        else:
            subgraph = graph

        # Define colors for different types of nodes
        color_map = []
        for node in subgraph:
            if isinstance(node, tuple) and node[0].startswith("§"):
                color_map.append('lightblue')  # sentences 
            elif isinstance(node, tuple) and node[0].isdigit() or 'Abschnitt' or 'Section' in subgraph.nodes[node]['label']:
                color_map.append('lightgreen')  # sections
            elif 'Zeichen' or 'Sign' in subgraph.nodes[node]['label']:
                color_map.append('pink') # signs
            elif 'Anlage' or 'Annex' in subgraph.nodes[node]['label']:
                color_map.append('purple') # tables
            elif '§' in subgraph.nodes[node]['label']:
                color_map.append('yellow') # paragraphs
            elif re.match(r"^[IVX]+\..*", subgraph.nodes[node]['label']):
                color_map.append('orange')
            else:
                color_map.append('gray')
        

        # Set the layout for the nodes in the graph
        pos = nx.spring_layout(subgraph, scale=2)

        # Create a plot with the specified size
        plt.figure(figsize=figsize)

        # Draw the graph with node color mapping
        nx.draw(subgraph, pos, with_labels=True, node_color=color_map, edge_color='gray', 
                node_size=500, font_size=12, font_weight='bold', arrowsize=20, arrowstyle='->', width=1.5)
        
        # change arrow length 
        
        # display legend only for the exisiting colors
        color_legend = {}
        for color in color_map:
            if color not in color_legend:
                if color == 'lightblue':
                    color_legend[color] = 'Absatz/subunit'
                elif color == 'lightgreen':
                    color_legend[color] = 'Abschnitt/section'
                elif color == 'pink':
                    color_legend[color] = 'Zeichen/Sign'
                elif color == 'purple':
                    color_legend[color] = 'Anlage/Annex'
                elif color == 'yellow':
                    color_legend[color] = 'Paragraph'
                elif color == 'orange':
                    color_legend[color] = 'Kategorie/category'
                elif color == 'gray':
                    color_legend[color] = 'Andere'

        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12, label=label) for color, label in color_legend.items()],
                   loc='upper right', bbox_to_anchor=(1, 1), fontsize=20)
        
        # Set the title of the plot
        plt.title(title)

        # Save the plot to a file
        plt.savefig(filename)

        
    
    def get_node_content_by_id(self, graph, node_id):
        return graph.nodes[node_id]['label']
    
    def get_nodes_connected_to_node(self, graph, node_id):
        return list(graph.neighbors(node_id))
    
    def get_content_of_connected_nodes(self, graph, node_id):
        connected_nodes = self.get_nodes_connected_to_node(graph, node_id)
        connected_nodes_content = []
        for connected_node in connected_nodes:
            connected_nodes_content.append(graph.nodes[connected_node]['label'])
        # add the content of the node itself at the beginning of the list
        connected_nodes_content.insert(0, graph.nodes[node_id]['label'])
        
        return connected_nodes_content
        
    ####################################### STRUCRTURE GRAPH ########################################
    def evaluate_graph(self,graph):
        """
        Evaluates the complexity of the given graph based on several metrics.
        
        :param graph: A NetworkX graph object.
        :return: A dictionary containing evaluation metrics.
        """
        # Number of nodes
        num_nodes = graph.number_of_nodes()
        
        # Number of edges
        num_edges = graph.number_of_edges()
        
        # Average branching factor
        avg_branching_factor = num_edges / num_nodes if num_nodes > 0 else 0
        
        # Out-degree for each node
        out_degrees = [degree for _, degree in graph.out_degree()]
        
        # Minimum and maximum out-degree
        min_out_degree = min(out_degrees) if out_degrees else 0
        max_out_degree = max(out_degrees) if out_degrees else 0
        
        # Create the evaluation report
        evaluation = {
            "Number of Nodes": num_nodes,
            "Number of Edges": num_edges,
            "Average Branching Factor": avg_branching_factor,
            "Minimum Out-Degree": min_out_degree,
            "Maximum Out-Degree": max_out_degree,
        }
        
        return evaluation
    
    def log_graph_structure(self, graph):
        """
        Goes through the integrated graph to log nodes and edges in the specified structure.

        :param graph: A NetworkX DiGraph object representing the integrated graph.
        :return: A dictionary where keys are node IDs and values are dictionaries
                 containing the node content and its neighbors with their content.
        """
        graph_log = {}
        for node_id in graph.nodes:
            node_content = graph.nodes[node_id].get('label', 'No Content')
            neighbors_data = {}
            for neighbor_id in graph.successors(node_id):
                neighbor_content = graph.nodes[neighbor_id].get('label', 'No Content')
                neighbors_data[str(neighbor_id)] = {'content': neighbor_content} # Convert neighbor_id to string

            graph_log[str(node_id)] = { # Convert node_id to string
                'content': node_content,
                'neighbours': neighbors_data
            }

        
        return graph_log

    # Node Edge description 
    def graph_to_text(self, graph: nx.Graph) -> List[str]:
        texts = []
        for node_id, data in graph.nodes(data=True):
            texts.append(f"Node: {node_id} - {data.get('label', 'No Label')}")
            for neighbor in graph.neighbors(node_id):
                texts.append(f"Edge: {node_id} -> {neighbor}")
        return texts

    # Enumerate paths
    def enumerate_paths(self,graph: nx.Graph, max_length: int) -> List[str]:
        """
        Extracts paths of a certain length from the graph and represents them
        as sequences of node descriptions.

        Args:
            graph: The NetworkX graph.
            max_length: The maximum length of the paths to extract (number of nodes).

        Returns:
            A list of strings, where each string represents a path.
        """
        paths = []
        for start_node in graph.nodes:
            for path in self._find_all_paths(graph, start_node, max_length):
                node_descriptions = [f"{node} - {graph.nodes[node].get('label', 'No Label')}" for node in path]
                path_string = " -> ".join(node_descriptions)
                paths.append(f"Path: {path_string}")
        return paths

    def _find_all_paths(self,graph: nx.Graph, start_node, max_length, current_path=None):
        """
        Recursive helper function to find all paths from a start node up to a
        specified maximum length without revisiting nodes in the current path.
        """
        if current_path is None:
            current_path = [start_node]
        else:
            current_path = current_path + [start_node]

        if len(current_path) > max_length:
            return []
        if len(current_path) > 1:
            yield current_path

        for neighbor in graph.neighbors(start_node):
            if neighbor not in current_path:
                yield from self._find_all_paths(graph, neighbor, max_length, current_path)
    
    def to_dot_language(self, graph, graph_name="KnowledgeGraph"):
        """
        Converts a networkx.DiGraph object into a DOT language string.

        Args:
            graph (networkx.DiGraph): The graph to convert.
            graph_name (str): The name of the graph for the DOT output.

        Returns:
            str: A string representing the graph in DOT language.
        """
        dot_string = f'digraph "{graph_name}" {{\n'
        dot_string += '  rankdir=LR;\n' # Optional: left-to-right layout

        # Add nodes
        for node, data in graph.nodes(data=True):
            node_label = data.get('label', str(node))
            # Escape double quotes in labels
            node_label = node_label.replace('"', '\\"')
            node_id = node #sanitize_node_id(node)

            # Optional: Add styling based on node type
            node_style = ""
            node_type = data.get('type')
            if node_type == "category":
                node_style = '  shape=box, style="filled", fillcolor="#ADD8E6", fontcolor="#333333"' # Light Blue
            elif node_type == "paragraph":
                node_style = '  shape=ellipse, style="filled", fillcolor="#90EE90", fontcolor="#333333"' # Light Green
            elif node_type == "sentence":
                node_style = '  shape=circle, style="filled", fillcolor="#FFD700", fontcolor="#333333"' # Gold
            elif node_type == "table":
                node_style = '  shape=folder, style="filled", fillcolor="#FFB6C1", fontcolor="#333333"' # Light Pink
            elif node_type == "section":
                node_style = '  shape=note, style="filled", fillcolor="#DA70D6", fontcolor="#FFFFFF"' # Orchid
            elif node_type == "sign":
                node_style = '  shape=hexagon, style="filled", fillcolor="#87CEEB", fontcolor="#333333"' # Sky Blue

            dot_string += f' ID {node_id} [content="{node_label}"{node_style}];\n'

        # Add edges
        for u, v in graph.edges():
            sanitized_u = u #sanitize_node_id(u)
            sanitized_v = v #sanitize_node_id(v)
            dot_string += f'  {sanitized_u} -> {sanitized_v};\n'

        dot_string += '}\n'
        return dot_string
    
    def graph_to_text_structured(self,main_graph, table_graph):
        """
        Converts the knowledge graph to a structured text format.

        Args:
            main_graph: The main knowledge graph (NetworkX DiGraph).
            table_graph: The table knowledge graph (NetworkX DiGraph).

        Returns:
            A string representing the graph in a structured text format.
        """
        documents = []

        def add_node_to_text(node_id, graph, tag_name, text):
            label = graph.nodes[node_id].get('label', '')
            return f"<{tag_name} id=\"{node_id}\" label=\"{label}\">{text}</{tag_name}>"

        # Process main graph categories
        for category_id in main_graph.nodes:
            if not isinstance(category_id, tuple):  # Assuming categories are not tuples
                category_text = add_node_to_text(category_id, main_graph, 'category', '')
                paragraph_texts = []
                for paragraph_id in main_graph.successors(category_id):
                    paragraph_text = add_node_to_text(paragraph_id, main_graph, 'paragraph', '')
                    sentence_texts = []
                    for sentence_id in main_graph.successors(paragraph_id):
                        sentence_text = add_node_to_text(sentence_id, main_graph, 'sentence', main_graph.nodes[sentence_id].get('label', ''))
                        sentence_texts.append(sentence_text)
                    paragraph_text += ''.join(sentence_texts)
                    paragraph_texts.append(paragraph_text)
                category_text += ''.join(paragraph_texts)

                #add tables
                for table_id in table_graph.nodes:
                    if not isinstance(table_id, tuple):
                        table_text = add_node_to_text(table_id, table_graph, 'table', '')
                        section_texts = []
                        for section_id in table_graph.successors(table_id):
                            section_text = add_node_to_text(section_id, table_graph, 'section','')
                            sign_texts = []
                            for sign_id in table_graph.successors(section_id):
                                sign_text = add_node_to_text(sign_id, table_graph, 'sign', table_graph.nodes[sign_id].get('label', ''))
                                sign_texts.append(sign_text)
                            section_text += ''.join(sign_texts)
                            section_texts.append(section_text)
                        table_text += ''.join(section_texts)
                        category_text += table_text
                documents.append(f"<doc>{category_text}</doc>")
        return '\n'.join(documents)
    
    # Triples generation
    def _node_id_to_string(self, node_id):
        """
        Converts a node ID (which can be a string or a tuple) into a unique string
        representation suitable for KGE model input.
        """
        if isinstance(node_id, tuple):
            # Example: ('paragraph_A', 'sentence_1') -> "paragraph_A__sentence_1"
            return "__".join(str(part) for part in node_id)
        return str(node_id)
    
    def _get_node_content(self, graph, node_id):
        """
        Retrieves the content of a node in the graph.
        """
        if node_id in graph.nodes:
            return graph.nodes[node_id].get('label', 'No Content')
        return 'No Content'

    def graph_to_triples(self, graph):
        """
        Converts a NetworkX graph with 'relation' attributes on edges
        into a list of (head_entity_str, relation_str, tail_entity_str) triples.
        """
        triples = []
        if graph is None:
            return triples
            
        for u, v, data in graph.edges(data=True):
            # Default relation if not specified on the edge (should ideally always be specified)
            relation = data.get('relation', 'unknown_relation')
            

            # Get the content of the head and tail nodes
            head_entity_content = self._get_node_content(graph, u)
            tail_entity_content = self._get_node_content(graph, v)
            
            # triples.append((head_entity_str, str(relation), tail_entity_str))
            triples.append((head_entity_content, str(relation), tail_entity_content))
            
        return triples
    

    def _replace_citation_tags(self, text, graph):
        """
        Replaces citation tags in the text with their corresponding content from the graph.
        """
        # Extract citation tags
        tags = self._extract_text_ids_from_tags(text)
        tags = {
            'sign': tags['sign_ids'],
            'par': tags['paragraph_ids'],
            'sen': tags['sentence_pairs'],
            'tab': tags['table_ids'],
            'sec': tags['section_pairs']
        }
        print(tags)
        
        # Replace each tag with its corresponding content from the graph
        for tag_type, tag_ids in tags.items():
            for tag_id in tag_ids:
                if tag_id in graph.nodes:
                    node_content = graph.nodes[tag_id].get('label', '')
                    text = text.replace(f"<{tag_type}>{tag_id}</{tag_type}>", node_content)
        
        return text

    def graph_to_sentence_classification_dataset(self, graph): # graph is passed as an argument
        """
        Extracts sentence nodes from the graph, processes their content to replace
        citation tags with actual content from the graph, and labels them with
        their parent paragraph ID.
        
        Returns:
            list: A list of dictionaries, each with "text" and "label" keys.
        """
        dataset = []
        for node_id in graph.nodes:
            if isinstance(node_id, tuple):
                # Extract the content of the node
                node_content = graph.nodes[node_id].get('label', '')
                
                # Replace citation tags with actual content from the graph
                # This is a placeholder; you need to implement the logic to replace tags
                # For example, if the node content contains <par> tags, replace them with actual paragraph content
                node_content = self._replace_citation_tags(node_content, graph)
                
                # Create a dictionary for the dataset entry
                dataset_entry = {
                    "text": node_content,
                    "label": str(node_id[0])  # Assuming the label is the first part of the tuple
                }
                dataset.append(dataset_entry)
        
        return dataset