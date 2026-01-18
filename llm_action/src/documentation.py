import os
import json

from bs4 import BeautifulSoup, Tag

from llm_action.src.config import MLIR_TRANSFORM_DIALECT_DOCS_URL

from llm_action.src.utils.scrape import fetch_html, main_container, blocks, build_tree, linear_tags, prune_at_level
from llm_action.src.utils.persistence import save_documentation, save_documentation_outline, save_documentation_tree_md, save_documentation_doc_md
from llm_action.src.models import TransformationDocumentation, TransformationCategory, Documentation

def load_documentation() -> Documentation:
    with open("llm_action/resources/ready/documentation.json", "r", encoding="utf-8") as f:
        documentation = json.load(f)
    return documentation

if __name__ == "__main__":

    out_dir = "llm_action/resources"

    html = fetch_html(MLIR_TRANSFORM_DIALECT_DOCS_URL)
    soup = BeautifulSoup(html, "html.parser")

    tags = linear_tags(main_container(soup))
    flat = blocks(tags)
    tree = build_tree(flat)

    tree_path = save_documentation(tree, os.path.join(out_dir, "raw"), "documentation.json")
    tree_md = save_documentation_tree_md(tree, os.path.join(out_dir, "raw"), "documentation.md")
    outline_path = save_documentation_outline(tree, os.path.join(out_dir, "raw"), "outline.txt")

    pruned_tree = prune_at_level(tree, target_level=2, depth=1)

    # "Transform Dialect" -> 
    transform_documentation = pruned_tree.children[1]
    transform_documentation.children = transform_documentation.children[8:-8]

    pruned_json = save_documentation(transform_documentation, os.path.join(out_dir, "processed"), "documentation.json")
    pruned_md = save_documentation_tree_md(transform_documentation, os.path.join(out_dir, "processed"), "documentation.md")
    pruned_outline = save_documentation_outline(transform_documentation, os.path.join(out_dir, "processed"), "outline.txt")

    transformation_categories = []
    for category_node in transform_documentation.children:
        transformations = []
        for transform_node in category_node.children:
            transformations.append(
                TransformationDocumentation(
                    name=transform_node.name[: transform_node.name.find("(")].strip(),
                    label=transform_node.name[transform_node.name.find("(")+1 : transform_node.name.find(")")].strip(),
                    content=transform_node.content,
                )
            )
        transformation_categories.append(
            TransformationCategory(
                name=category_node.name[: category_node.name.find("¶")].strip(),
                transformations=transformations,
            )
        )
    documentation = Documentation(transformation_categories=transformation_categories)
    
    documentation_lookup = {}
    for category in documentation.transformation_categories:
        documentation_lookup[category.name] = {}
        for transformation in category.transformations:
            documentation_lookup[category.name][transformation.name] = transformation.content

    with open(os.path.join(out_dir, "ready", "documentation.json"), "w", encoding="utf-8") as f:
        json.dump(documentation_lookup, f, indent=4)

    save_documentation_doc_md(documentation, os.path.join(out_dir, "ready"), "documentation.md")

    representation = ""
    for category in documentation.transformation_categories:
        representation += f"{category.name}\n"
        for transformation in category.transformations:
            representation += f"- {transformation.name}\n"
            
    with open(os.path.join(out_dir, "ready", "representation.txt"), "w", encoding="utf-8") as f:
        f.write(representation)

    print("Wrote:")
    print(f" - Full documentation JSON: {tree_path}")
    print(f" - Full documentation MD: {tree_md}")
    print(f" - Full outline: {outline_path}")
    print(f" - Pruned documentation JSON: {pruned_json}")
    print(f" - Pruned documentation MD: {pruned_md}")
    print(f" - Pruned outline: {pruned_outline}")
    print(f" - Ready documentation JSON: {os.path.join(out_dir, 'ready', 'documentation.json')}")
    print(f" - Ready documentation MD: {os.path.join(out_dir, 'ready', 'documentation.md')}")
    print(f" - Ready representation TXT: {os.path.join(out_dir, 'ready', 'representation.txt')}")
