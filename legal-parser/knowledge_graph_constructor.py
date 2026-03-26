import html
import json
import random
import networkx as nx
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Any, Dict, List
from functools import partial   
import os


class KnowledgeGraphConstructor:
    def __init__(self, main_data, table_data):
        self.main_data = main_data
        self.table_data = table_data

    def main_knowledge_graph(self):
        main_graph = nx.DiGraph()
        for category_data in self.main_data:
            category_node_id = category_data['category']
            main_graph.add_node(category_node_id, label = category_data['category'], id = category_node_id, type = 'category')
            for paragraph in category_data['paragraphs']:
                paragraph_node_id = paragraph['paragraph_id']
                paragraph_node_content = paragraph['paragraph']
                main_graph.add_node(paragraph_node_id, label = paragraph_node_content , id = paragraph_node_id, type = 'paragraph')
                main_graph.add_edge(category_node_id, paragraph_node_id, relation = 'has_paragraph')
                main_graph.nodes[category_node_id][paragraph_node_id] = paragraph_node_content
                for sentence_dict in paragraph['sentences']:
                    for sen_id, sen_text in sentence_dict.items():
                        sen_node_id = (paragraph_node_id, sen_id) # f'{paragraph_node_name}:{sen_id}'
                        sen_node_content = sen_text
                        main_graph.add_node(sen_node_id, label = sen_node_content, id = sen_node_id, type = 'sentence')
                        main_graph.add_edge(paragraph_node_id, sen_node_id, relation = 'has_sentence')
                        main_graph.nodes[paragraph_node_id][sen_node_id] = sen_node_content
                    
        return main_graph
    
  

    def table_knowledge_graph(self):
        table_graph = nx.DiGraph()
        # Parse through each table
        for table in self.table_data:
            table_node_id = table['table_id']
            table_node_content = table['table_name']
            table_graph.add_node(table_node_id, label=table_node_content, id=table_node_id, type = 'table')
            table_graph.nodes[table_node_id][table_node_id] = table_node_content
            # Parse through each section in the table
            for section in table['table_data']:
                section_id = section['Section_Id']
                section_node_id = (table_node_id,section_id)
                section_node_content = section['Section']
                table_graph.add_node(section_node_id, label=section_node_content, id=section_node_id, type = 'section')
                table_graph.add_edge(table_node_id, section_node_id, relation = 'has_section')
                table_graph.nodes[table_node_id][section_node_id] = section_node_content
                # Parse through each content in the section
                for content in section['Content']:
                    sign_id = content['Sign_id']
                    sign_number = content['Sign_number']
                    sign_name = content['Sign_name']
                    image = content['image']
                    description = content['description']


                    sign_node_id =  sign_number #sign_name
                    sign_node_content = f'{sign_name} - {description} - {image}'
                    table_graph.add_node(sign_node_id, label=sign_node_content, id=sign_node_id, type = 'sign')
                    table_graph.add_edge(section_node_id, sign_node_id, relation = 'has_sign')
                    table_graph.nodes[section_node_id][sign_node_id] = sign_node_content

        return table_graph
    
    def _extract_image_ids_from_tags(self,text):
        """
        Extracts image IDs from `<img>` tags within a given text.

        :param text: The text string to search for image tags.
        :return: A list of image IDs extracted from the tags.
        """
        # Regex pattern to capture content within <img> tags
        img_pattern = re.compile(r'<img>(.*?)</img>')
        
        # Finding all occurrences of the pattern
        image_ids = img_pattern.findall(text)

        # clean text from image tags
        cleaned_text = re.sub(r'<img>.*?</img>', '', text)
        
        return image_ids , cleaned_text

    def _extract_text_ids_from_tags(self,text):
        # Regex patterns to capture various references
        sign_pattern = re.compile(r'<sign>(.*?)</sign>')
        par_pattern = re.compile(r'<par>(§\s*\d+)</par>')
        # par_pattern = re.compile(r'(?:<par>)?(§\s*\d+)</par>')
        sen_pattern = re.compile(r'<sen>(.*?)</sen>')
        tab_pattern = re.compile(r'<tab>(.*?)</tab>')
        sec_pattern = re.compile(r'<sec>(.*?)</sec>')

        # Initialize structures to collect data
        signs = sign_pattern.findall(text)
        paragraphs = par_pattern.findall(text)
        sentences = sen_pattern.findall(text)
        tables = tab_pattern.findall(text)
        sections = sec_pattern.findall(text)

        # Establish relationships between paragraphs and sentences, tables and sections
        paragraph_to_sentences = {}
        table_to_sections = {}

        current_paragraph = None
        current_table = None

        # Split and process segments
        segments = re.split(r'(<par>.*?</par>|<sen>.*?</sen>|<tab>.*?</tab>|<sec>.*?</sec>)', text)
        for segment in segments:
            par_match = par_pattern.search(segment)
            sen_match = sen_pattern.search(segment)
            tab_match = tab_pattern.search(segment)
            sec_match = sec_pattern.search(segment)

            if par_match:
                current_paragraph = par_match.group(1)
                paragraph_to_sentences[current_paragraph] = []

            elif sen_match and current_paragraph:
                paragraph_to_sentences[current_paragraph].append(sen_match.group(1))

            if tab_match:
                current_table = tab_match.group(1)
                table_to_sections[current_table] = []

            elif sec_match and current_table:
                table_to_sections[current_table].append(sec_match.group(1))

        # Filter out paragraphs and tables that are linked to sentences and sections
        linked_paragraphs = [par for par, sens in paragraph_to_sentences.items() if sens]
        linked_tables = [tab for tab, secs in table_to_sections.items() if secs]

        standalone_paragraphs = [par for par in paragraphs if par not in linked_paragraphs]
        standalone_tables = [tab for tab in tables if tab not in linked_tables]

        
        tags =  {
            'sign_ids': signs,
            'paragraph_ids': standalone_paragraphs,
            'table_ids': standalone_tables,
            'sentence_pairs': [(par, sen) for par, sens in paragraph_to_sentences.items() for sen in sens], # paragraph vs absatz 
            'section_pairs': [(tab, sec) for tab, secs in table_to_sections.items() for sec in secs]
        }
        
        # check the returned ids if they contain any pattern as well call the function recursively
        # text = '(<par><par>§ 36</par>, <par>§ 37</par>, <par>§ 38</par>, <par>§ 39</par>, <par>§ 40</par>, <par>§ 41</par>, <par>§ 42</par>, <par>§ 43</par></par> in Verbindung mit den Anlage <tab>1</tab>, Anlage <tab>2</tab>, Anlage <tab>3</tab>, Anlage <tab>4</tab>)'
        # {'sign_ids': [], 'paragraph_ids': ['§ 36', '§ 37', '§ 38', '§ 39', '§ 40', '§ 41', '§ 42', '§ 43'], 'table_ids': ['1', '2', '3', '4'], 'sentence_pairs': [], 'section_pairs': []}
        
        return tags

    def connect_nodes_based_on_text(self,main_graph, table_graph):
        # Create a single integrated graph that includes both main and table graphs
        integrated_graph = nx.union(main_graph, table_graph)
        
        # Iterate over all sentence nodes in the main graph
        for node in main_graph.nodes:
            if isinstance(node, tuple):
                sentence_content = main_graph.nodes[node]['label']
                tags = self._extract_text_ids_from_tags(sentence_content)
                
                # print(f"Sentence: {sentence_content}")
                # print(f"Tags: {tags}")

                sign_ids = tags['sign_ids']
                paragraph_node_ids = tags['paragraph_ids']
                sentence_node_pairs = tags['sentence_pairs']
                table_node_ids = tags['table_ids']
                section_node_pairs = tags['section_pairs']

                
                # Connect the sentence node to the paragraph nodes
                for paragraph_id in paragraph_node_ids:
                    if paragraph_id in main_graph.nodes:
                        integrated_graph.add_edge(node, paragraph_id, relation = 'has_paragraph')
                        # print(f"Connected {node} to paragraph {paragraph_id}")
                        

                
                # Connect the sentence node to the sentence nodes
                for sentence_pair in sentence_node_pairs:
                    if sentence_pair in main_graph.nodes:
                        integrated_graph.add_edge(node, sentence_pair, relation = 'has_sentence')
                        # print(f"Connected {node} to sentence {sentence_pair}")


                # Connect the sentence node to the sign nodes
                for sign_id in sign_ids:
                    if sign_id in table_graph.nodes:
                        integrated_graph.add_edge(node, sign_id, relation = 'has_sign')
                        # print(f"Connected {node} to  sign {sign_id}")

                # Connect the sentence node to the table nodes
                for table_id in table_node_ids:
                    if table_id in table_graph.nodes:
                        integrated_graph.add_edge(node, table_id, relation = 'has_table')
                        # print(f"Connected {node} to table {table_id}")

                # Connect the sentence node to the section nodes
                for section_pair in section_node_pairs:
                    if section_pair in table_graph.nodes:
                        integrated_graph.add_edge(node, section_pair, relation = 'has_section')
                        # print(f"Connected {node} to  section {section_pair}")

        # iterate over all sentence nodes in the table graph (sign nodes)    
        for node in table_graph.nodes:
            # check tables nodes in the table graph
            if isinstance(node, str) and 'Anlage' or 'Annex' in table_graph.nodes[node]['label']:
                table_table_content = table_graph.nodes[node]['label']
                tags = self._extract_text_ids_from_tags(table_table_content)
                sign_ids = tags['sign_ids']
                paragraph_node_ids = tags['paragraph_ids']
                sentence_node_pairs = tags['sentence_pairs']
                table_node_ids = tags['table_ids']
                section_node_pairs = tags['section_pairs']

                # Connect the table node to the paragraph nodes
                for paragraph_id in paragraph_node_ids:
                    if paragraph_id in main_graph.nodes:
                        integrated_graph.add_edge(node, paragraph_id, relation = 'has_paragraph')
                        # print(f"Connected {node} to paragraph {paragraph_id}")

                # Connect the table node to the sentence nodes
                for sentence_pair in sentence_node_pairs:
                    if sentence_pair in main_graph.nodes:
                        integrated_graph.add_edge(node, sentence_pair, relation = 'has_sentence')
                        # print(f"Connected {node} to sentence {sentence_pair}")

                # Connect the table node to the table nodes
                for table_id in table_node_ids:
                    if table_id in table_graph.nodes:
                        integrated_graph.add_edge(node, table_id, relation = 'has_table')
                        # print(f"Connected {node} to table {table_id}")

                # Connect the table node to the section nodes
                for section_pair in section_node_pairs:
                    if section_pair in table_graph.nodes:
                        integrated_graph.add_edge(node, section_pair, relation = 'has_section')
                        # print(f"Connected {node} to section {section_pair}")

                # connect the table node to other sign nodes
                for sign_id in sign_ids:
                    if sign_id in table_graph.nodes:
                        integrated_graph.add_edge(node, sign_id, relation = 'has_sign')
                        # print(f"Connected {node} to sign {sign_id}")
            # Check signs nodes in the table graph
            elif isinstance(node, str) and 'Zeichen' or 'Sign' in table_graph.nodes[node]['label']:
                sign_table_content = table_graph.nodes[node]['label']
                tags = self._extract_text_ids_from_tags(sign_table_content)
                sign_ids = tags['sign_ids']
                paragraph_node_ids = tags['paragraph_ids']
                sentence_node_pairs = tags['sentence_pairs']
                table_node_ids = tags['table_ids']
                section_node_pairs = tags['section_pairs']

                # Connect the sign node to the paragraph nodes
                for paragraph_id in paragraph_node_ids:
                    if paragraph_id in main_graph.nodes:
                        integrated_graph.add_edge(node, paragraph_id, relation = 'has_paragraph')
                        # print(f"Connected {node} to paragraph {paragraph_id}")

                # Connect the sign node to the sentence nodes
                for sentence_pair in sentence_node_pairs:
                    if sentence_pair in main_graph.nodes:
                        integrated_graph.add_edge(node, sentence_pair, relation = 'has_sentence')
                        # print(f"Connected {node} to sentence {sentence_pair}")

                # Connect the sign node to the table nodes
                for table_id in table_node_ids:
                    if table_id in table_graph.nodes:
                        integrated_graph.add_edge(node, table_id, relation = 'has_table')
                        # print(f"Connected {node} to table {table_id}")

                # Connect the sign node to the section nodes
                for section_pair in section_node_pairs:
                    if section_pair in table_graph.nodes:
                        integrated_graph.add_edge(node, section_pair, relation = 'has_section')
                        # print(f"Connected {node} to section {section_pair}")

                # connect the sign node to other sign nodes
                for sign_id in sign_ids:
                    if sign_id in table_graph.nodes:
                        integrated_graph.add_edge(node, sign_id, relation = 'has_sign')
                        # print(f"Connected {node} to sign {sign_id}")
                        
            # check sections nodes in the table graph
            elif isinstance(node, tuple) and 'Abschnitt' or 'Section' in table_graph.nodes[node]['label']:
                section_table_content = table_graph.nodes[node]['label']
                tags = self._extract_text_ids_from_tags(section_table_content)
                # print(f"Section: {section_table_content}")
                # print(f"Tags: {tags}")
                sign_ids = tags['sign_ids']
                paragraph_node_ids = tags['paragraph_ids']
                sentence_node_pairs = tags['sentence_pairs']
                table_node_ids = tags['table_ids']
                section_node_pairs = tags['section_pairs']

                # Connect the section node to the paragraph nodes
                for paragraph_id in paragraph_node_ids:
                    if paragraph_id in main_graph.nodes:
                        integrated_graph.add_edge(node, paragraph_id, relation = 'has_paragraph')
                        # print(f"Connected {node} to paragraph {paragraph_id}")

                # Connect the section node to the sentence nodes
                for sentence_pair in sentence_node_pairs:
                    if sentence_pair in main_graph.nodes:
                        integrated_graph.add_edge(node, sentence_pair, relation = 'has_sentence')
                        # print(f"Connected {node} to sentence {sentence_pair}")

                # Connect the section node to the table nodes
                for table_id in table_node_ids:
                    if table_id in table_graph.nodes:
                        integrated_graph.add_edge(node, table_id, relation = 'has_table')
                        # print(f"Connected {node} to table {table_id}")

                # Connect the section node to the section nodes
                for section_pair in section_node_pairs:
                    if section_pair in table_graph.nodes:
                        integrated_graph.add_edge(node, section_pair, relation = 'has_section')
                        # print(f"Connected {node} to section {section_pair}")

                # connect the section node to other sign nodes
                for sign_id in sign_ids:
                    if sign_id in table_graph.nodes:
                        integrated_graph.add_edge(node, sign_id, relation = 'has_sign')
                        # print(f"Connected {node} to sign {sign_id}")
            
            

        return integrated_graph

            

    # --- This assumes the function is part of a class ---
    def draw_graph(self, graph, start_node=None, title='Graph Visualization', filename='./graph_plots/graph.png', figsize=(12, 12)):
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

        # Define a robust mapping of node types to colors and legend labels
        color_map_dict = {
            'sentence': 'lightblue',
            'section': 'lightgreen',
            'sign': 'pink',
            'annex': 'purple',
            'paragraph': 'yellow',
            'category': 'orange',
            'other': 'gray'
        }
        label_map_dict = {
            'lightblue': 'Absatz/subunit',
            'lightgreen': 'Abschnitt/section',
            'pink': 'Zeichen/Sign',
            'purple': 'Anlage/Annex',
            'yellow': 'Paragraph',
            'orange': 'Kategorie/category',
            'gray': 'Andere'
        }

        # Assign colors to each node with corrected logic (most specific checks first)
        node_colors = []
        for node in subgraph:
            # Safely get the label, defaulting to an empty string if it doesn't exist
            label = subgraph.nodes[node].get('label', '')
            
            # Use corrected boolean logic with explicit checks
            if 'Zeichen' in label or 'Sign' in label:
                node_colors.append(color_map_dict['sign'])
            elif 'Anlage' in label or 'Annex' in label:
                node_colors.append(color_map_dict['annex'])
            elif 'Abschnitt' in label or 'Section' in label:
                node_colors.append(color_map_dict['section'])
            elif '§' in label:
                node_colors.append(color_map_dict['paragraph'])
            elif re.match(r"^[IVX]+\..*", label):
                node_colors.append(color_map_dict['category'])
            # Specific check for nodes that are tuples starting with "§"
            elif isinstance(node, tuple) and len(node) > 0 and str(node[0]).startswith("§"):
                node_colors.append(color_map_dict['sentence'])
            else:
                node_colors.append(color_map_dict['other'])
        

        # Create a plot with the specified size
        plt.figure(figsize=figsize)

        # Set the layout for the nodes in the graph
        pos = nx.spring_layout(subgraph, scale=2)

        # Draw the graph with node color mapping
        nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, edge_color='gray', 
                node_size=500, font_size=12, font_weight='bold', arrowsize=20, arrowstyle='->', width=1.5)
        
        # Create legend handles only for the colors present in the graph
        used_colors = set(node_colors)
        legend_handles = []
        for color in used_colors:
            label = label_map_dict.get(color, 'Undefined')
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12, label=label)
            )
        
        # Display the legend
        plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Set the title of the plot
        plt.title(title)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the plot to a file
        plt.savefig(filename)

        plt.close() # Close the figure to prevent memory leaks
        
    
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
    
    def to_dot_language(self, graph, graph_name="KnowledgeGraph"):
        """
        Converts a networkx.DiGraph object into a DOT language string.
        Revised to properly escape characters for valid syntax.

        Args:
            graph (networkx.DiGraph): The graph to convert.
            graph_name (str): The name of the graph for the DOT output.

        Returns:
            str: A string representing the graph in valid DOT language.
        """
        dot_lines = [f'digraph "{graph_name}" {{'] # It's also good practice to quote the graph name
        
        # Add nodes with labels
        for node, data in graph.nodes(data=True):
            node_id = self._node_id_to_string(node)
            
            # --- KEY FIX: Escape double quotes in the label ---
            label_text = data.get('label', node_id)
            escaped_label = str(label_text).replace('"', '\\"')
            
            dot_lines.append(f'    "{node_id}" [label="{escaped_label}"];')
        
        # Add edges with relations
        for u, v, data in graph.edges(data=True):
            u_str = self._node_id_to_string(u)
            v_str = self._node_id_to_string(v)
            
            # --- KEY FIX: Escape double quotes in the relation ---
            relation_text = data.get('relation', 'related_to')
            escaped_relation = str(relation_text).replace('"', '\\"')
            
            dot_lines.append(f'    "{u_str}" -> "{v_str}" [label="{escaped_relation}"];')
        
        dot_lines.append('}')
        return "\n".join(dot_lines)

    
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
            if head_entity_content == 'No Content' or tail_entity_content == 'No Content':
                continue
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
        # print(tags)
        
        # Replace each tag with its corresponding content from the graph
        for tag_type, tag_ids in tags.items():
            for tag_id in tag_ids:
                if tag_id in graph.nodes:
                    node_content = graph.nodes[tag_id].get('label', '')
                    text = text.replace(f"<{tag_type}>{tag_id}</{tag_type}>", f'<{tag_type}>{node_content}</{tag_type}>')

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
                    # "label_id": str(node_id[0]),  # Assuming the label is the first part of the tuple else remove the node from the graph
                    "label": self._get_node_content(graph, node_id[0]) if len(node_id) > 0 else None
                }

                if dataset_entry["label"] is not None and dataset_entry["label"] != "No Content":
                    dataset.append(dataset_entry)

        return dataset

    # Assuming the function is part of the KnowledgeGraphConstructor class
    def graph_to_nsp_dataset(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """
        Prepares a dataset for Next Sentence Prediction (NSP) and Masked Language Modeling (MLM)
        by creating consecutive and non-consecutive sentence pairs from the graph.

        Args:
            graph (nx.Graph): The graph object, containing nodes for sentences and paragraphs.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a sentence pair
                                    and a label for NSP.
        """
        dataset = []
        
        # 1. Extract all sentence nodes from the graph
        sentence_nodes = [node for node in graph.nodes if graph.nodes[node].get('type') == 'sentence']
        
        # 2. Sort the sentence nodes to get them in the correct document order
        # This is a crucial step for correctly identifying consecutive sentences.
        # The node IDs are tuples like (paragraph_id, sentence_id).
        sentence_nodes.sort()
        
        # 3. Create a list of all sentence contents for generating non-consecutive pairs
        sentence_corpus = [self._get_node_content(graph, node) for node in sentence_nodes]
        
        # 4. Iterate through the sorted sentences to create pairs
        for i in range(len(sentence_nodes) - 1):
            # A. Create a consecutive pair (label 0)
            text_a = self._get_node_content(graph, sentence_nodes[i])
            text_b = self._get_node_content(graph, sentence_nodes[i+1])
            
            # Check if the sentences are in the same paragraph
            # This is an important check to ensure pairs are truly consecutive
            # replace citation tags
            text_a = self._replace_citation_tags(text_a, graph)
            text_b = self._replace_citation_tags(text_b, graph)
            if sentence_nodes[i][0] == sentence_nodes[i+1][0]:
                dataset.append({
                    "text_a": text_a,
                    "text_b": text_b,
                    "label": 1  # 1 indicates consecutive sentences
                })
            
            # B. Create a non-consecutive pair (label 1)
            # Select a random sentence from the corpus
            random_sentence = random.choice(sentence_corpus)
            while random_sentence == text_b:
                random_sentence = random.choice(sentence_corpus)

            # replace citation tags
            text_a = self._replace_citation_tags(text_a, graph)
            random_sentence = self._replace_citation_tags(random_sentence, graph)

            dataset.append({
                "text_a": text_a,
                "text_b": random_sentence,
                "label": 0  # 0 indicates non-consecutive sentences
            })
                
        return dataset

    def graph_to_raw_text(self, graph: nx.Graph) -> str:
        """
        Convert the knowledge graph into raw text format.

        Args:
            graph (nx.Graph): The graph object to convert.

        Returns:
            str: The raw text representation of the graph.
        """
        sentences = []
        for node in graph.nodes:
                sentences.append(self._get_node_content(graph, node))
        return "\n".join(sentences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Graph Constructor')
    parser.add_argument('--parsed_main', type=str , default=None, help='Directory to save the parsed main content')
    parser.add_argument('--parsed_table', type=str , default=None, help='Directory to save the parsed table content')
    parser.add_argument('--main_graph', type=str , default=None, help='Directory to save the main graph')
    parser.add_argument('--table_graph', type=str , default=None, help='Directory to save the table graph')
    parser.add_argument('--integrated_graph', type=str , default=None, help='Directory to save the integrated graph')
    parser.add_argument('--out_dir', type=str , default=None, help='Directory to save the output files')
    args = parser.parse_args()

    
    # read main content from json file
    try:
        with open(args.parsed_main, 'r') as f:
            main_content = json.load(f)
    except:
        main_content = []    
    # read table content from json file
    try :
        with open(args.parsed_table, 'r') as f:
            table_content = json.load(f)
    except:
        table_content = []

    # we can also take the translated data 

    knowledge_graph_constructor = KnowledgeGraphConstructor(main_content, table_content)

    main_graph = knowledge_graph_constructor.main_knowledge_graph()
    table_graph = knowledge_graph_constructor.table_knowledge_graph()

    if table_graph:
        integrated_graph = knowledge_graph_constructor.connect_nodes_based_on_text(main_graph, table_graph)
    
    else:
        integrated_graph = main_graph

    # remove invalid nodes from graph
    main_graph.remove_nodes_from([node for node in main_graph.nodes if not node])
    table_graph.remove_nodes_from([node for node in table_graph.nodes if not node])
    
    main_graph.remove_nodes_from([node for node in main_graph.nodes if main_graph.nodes[node].get('label') == "No Paragraph Title"])
    table_graph.remove_nodes_from([node for node in table_graph.nodes if table_graph.nodes[node].get('label') == "No Paragraph Title"])
    # remove 
    integrated_graph.remove_nodes_from([node for node in integrated_graph.nodes if not node])


    # draw first 20 for each graph
    # check existing of folders if not make
    os.makedirs("./graph_plots", exist_ok=True)

    # sub_main_graph = main_graph.subgraph(list(main_graph.nodes)[:30])
    # knowledge_graph_constructor.draw_graph(sub_main_graph, title='Main Knowledge Graph', filename= f"./graph_plots/{args.main_graph}.png")
    # sub_table_graph = table_graph.subgraph(list(table_graph.nodes)[:20])
    # knowledge_graph_constructor.draw_graph(sub_table_graph, title='Table Knowledge Graph', filename=f"./graph_plots/{args.table_graph}.png")
    # sub_integrated_graph = integrated_graph.subgraph(list(integrated_graph.nodes))
    # knowledge_graph_constructor.draw_graph(integrated_graph,start_node=(''),title='Integrated Knowledge Graph', filename=f"./graph_plots/{args.integrated_graph}.png")
    # knowledge_graph_constructor.draw_graph(sub_integrated_graph, title='Integrated Knowledge Graph', filename=f"./graph_plots/{args.integrated_graph}.png")

    # # extratct text ids (to be fixed example)
    text = '(<par><par>§ 36</par>, <par>§ 37</par>, <par>§ 38</par>, <par>§ 39</par>, <par>§ 40</par>, <par>§ 41</par>, <par>§ 42</par>, <par>§ 43</par></par> in Verbindung mit den Anlage <tab>1</tab>, Anlage <tab>2</tab>, Anlage <tab>3</tab>, Anlage <tab>4</tab>)'
    print(knowledge_graph_constructor._extract_text_ids_from_tags(text))

    
    # Example usage:
    integrated_graph_evaluation =knowledge_graph_constructor.evaluate_graph(integrated_graph)
    print("Graph Evaluation Metrics:")
    for metric, value in integrated_graph_evaluation.items():
        print(f"{metric}: {value}")


    ######################## FORMATTED KNOWLEDGE GRAPH ##############################
    
    
    # CLM : (1) convert graph to DOT Language
    dot_str = knowledge_graph_constructor.to_dot_language(integrated_graph, graph_name="KnowledgeGraph")
    with open(f'{args.out_dir}/{args.integrated_graph}.dot', 'w') as f:
        f.write(dot_str)

    #CLU : CLUSTERING
    # (2) convert graph to sentence classification dataset
    dataset = knowledge_graph_constructor.graph_to_sentence_classification_dataset(main_graph)
    with open(f'{args.out_dir}/{args.main_graph}_sentence_clu_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    # MLM : Next sentence prediction task (NSP)
    mlm_dataset = knowledge_graph_constructor.graph_to_nsp_dataset(main_graph)
    with open(f'{args.out_dir}/{args.main_graph}_nsp_dataset.json', 'w') as f:
        json.dump(mlm_dataset, f, indent=4, ensure_ascii=False)

    # Paragraph Embeddings 
    # (4) convert graph to triples
    triples = knowledge_graph_constructor.graph_to_triples(integrated_graph)
    with open(f'{args.out_dir}/{args.integrated_graph}_triples.txt', 'w') as f:
        for triple in triples:
            f.write(f"{triple[0]} <tr> {triple[1]} <tr> {triple[2]}\n")

    #(5) Convert graph to raw text
    raw_text = knowledge_graph_constructor.graph_to_raw_text(integrated_graph)
    with open(f'{args.out_dir}/{args.integrated_graph}_raw.txt', 'w') as f:
        f.write(raw_text)



    


