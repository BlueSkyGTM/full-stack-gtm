#!/usr/bin/env python3
"""
Fix graphify community labels with semantic names based on file paths.
Maps communities to meaningful phase/topic names instead of "Community N".
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

GRAPH_DIR = Path(__file__).parent.parent / "graphify-out"
LABELS_FILE = GRAPH_DIR / ".graphify_labels.json"
GRAPH_FILE = GRAPH_DIR / "graph.json"


def load_graph():
    """Load the graph JSON."""
    with open(GRAPH_FILE) as f:
        return json.load(f)


def get_phase_from_path(path):
    """Extract phase number and name from file path."""
    phases = {
        "00": "Setup & Tooling",
        "01": "Math Foundations",
        "02": "ML Fundamentals",
        "03": "Deep Learning Core",
        "04": "Computer Vision",
        "05": "NLP Foundations to Advanced",
        "06": "Speech & Audio",
        "07": "Transformers Deep Dive",
        "08": "Generative AI",
        "09": "Reinforcement Learning",
        "10": "LLMs From Scratch",
        "11": "LLM Engineering",
        "12": "Multimodal AI",
        "13": "Tools & Protocols",
        "14": "Agent Engineering",
        "15": "Autonomous Systems",
        "16": "Multi-Agent & Swarms",
        "17": "Infrastructure & Production",
        "18": "Ethics Safety Alignment",
        "19": "Capstone Projects",
    }
    
    # Match phase pattern: phases/NN-xxx
    match = re.search(r'phases/(\d+)-', path)
    if match:
        phase_num = match.group(1)
        return phases.get(phase_num, f"Phase {phase_num}")
    
    # Infrastructure
    if any(x in path for x in ["site/", "progress/", "glossary/", "assets/", "outputs/"]):
        return "Infrastructure"
    
    # Scripts
    if "scripts/" in path:
        return "Scripts & Utilities"
    
    # Quiz factory
    if "quiz-factory/" in path:
        return "Quiz Factory"
    
    # Skills
    if ".claude/skills/" in path or ".agents/skills/" in path:
        return "Agent Skills"
    
    # Config
    if any(x in path for x in ["vercel.json", "README.md", "ROADMAP.md", "catalog.json"]):
        return "Config & Docs"
    
    return "Other"


def analyze_community_nodes(graph):
    """Analyze what types of files are in each community."""
    community_files = {}
    
    for node in graph["nodes"]:
        comm_id = node.get("community", -1)
        if comm_id not in community_files:
            community_files[comm_id] = []
        community_files[comm_id].append(node.get("file_path", ""))
    
    return community_files


def generate_semantic_labels(community_files):
    """Generate semantic labels based on file distributions."""
    labels = {}
    
    for comm_id, files in community_files.items():
        # Count phase/topic occurrences
        topics = Counter(get_phase_from_path(f) for f in files)
        
        # Get dominant topic
        if topics:
            dominant_topic, count = topics.most_common(1)[0]
            percentage = count / len(files)
            
            # If >70% of files are in one phase/topic, use that name
            if percentage > 0.7:
                labels[comm_id] = dominant_topic
            else:
                # Mix of topics - create compound label
                top_topics = [t for t, _ in topics.most_common(2)]
                labels[comm_id] = " + ".join(top_topics)
        else:
            labels[comm_id] = f"Community {comm_id}"
    
    return labels


def fix_labels():
    """Main function to fix graphify labels."""
    print("Loading graph...")
    graph = load_graph()
    
    print("Analyzing communities...")
    community_files = analyze_community_nodes(graph)
    print(f"Found {len(community_files)} communities")
    
    print("Generating semantic labels...")
    labels = generate_semantic_labels(community_files)
    
    # Add any missing communities
    max_comm = max(graph["nodes"], key=lambda n: n.get("community", -1)).get("community", -1)
    for i in range(max_comm + 1):
        if i not in labels:
            labels[i] = f"Community {i}"
    
    # Save new labels
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    
    print(f"\n✓ Updated {len(labels)} community labels")
    print("\nSample semantic labels:")
    for comm_id in sorted(list(labels.keys())[:15]):
        print(f"  {comm_id}: {labels[comm_id]}")
    
    # Show distribution
    print("\nLabel distribution:")
    label_counts = Counter(labels.values())
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count} communities")


if __name__ == "__main__":
    fix_labels()