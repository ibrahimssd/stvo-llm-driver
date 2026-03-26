from graphviz import Source

# DOT string
dot_code = """
digraph ParserWorkflow {
    rankdir=LR;
    node [shape=rectangle, style=filled, color=lightblue, fontname=Helvetica];

    Pi [label="Parser (\u03A0)"];
    Psi [label="Predefined Syntax Patterns (\u03A8)"];
    P1 [label="Part (\u03A0\u2081)"];
    P2 [label="Part (\u03A0\u2082)"];
    Xi [label="Entities Extraction (\u039E)"];
    T [label="Building a Tree (\u1D31)"];
    Theta [label="Entity Tagging (\u0398)"];
    Phi [label="Entity Translation (\u03A6) (Optional)"];
    KG [label="Knowledge Graph Construction (\u1D4A)"];

    subgraph cluster_extraction {
        label="Entities Extracted (\u03B5)";
        style=dotted;
        AB [label="Absätze (\u1D2C\u1D2B)"];
        C [label="Categories (\u1D4B)"];
        P [label="Paragraphs (\u1D4F)"];
        S [label="Sentences (\u1D4E)"];
        CH [label="Abschnitt (\u1D2C\u1D39)"];
        A [label="Appendices (\u1D2C\u1D00\u2099)"];
    }

    Pi -> Psi [label="Customization"];
    Psi -> P1 [label="Parse"];
    Psi -> P2 [label="Parse"];
    P1 -> Xi;
    P2 -> Xi;
    Xi -> T [label="Hierarchical Syntax"];
    T -> Theta [label="Tag Nodes"];
    Theta -> Phi [label="Optional Translation"];
    Phi -> KG [label="Cross-Lingual Usability"];
    Theta -> KG [label="Tree-to-KG Conversion"];
    
    Xi -> AB [label="Extract"];
    Xi -> C [label="Extract"];
    Xi -> P [label="Extract"];
    Xi -> S [label="Extract"];
    Xi -> CH [label="Extract"];
    Xi -> A [label="Extract"];
}
"""

# Render the DOT graph
graph = Source(dot_code)
graph.render('workflow', format='png', cleanup=True)  # Save as workflow.png
print("Graph has been rendered as 'workflow.png'")
