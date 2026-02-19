
import os
import markdown
import sys

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simple conversion with some basic CSS for research paper look
    content = markdown.markdown(text, extensions=['extra', 'codehilite', 'toc'])
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Research Paper - SAM-AI</title>
        <style>
            body {{
                font-family: 'Times New Roman', Times, serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px;
                background: #f4f4f4;
            }}
            .container {{
                background: white;
                padding: 50px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            pre {{ background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            code {{ font-family: 'Courier New', Courier, monospace; background: #eee; padding: 2px 4px; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            blockquote {{ border-left: 5px solid #ccc; margin: 20px 0; padding-left: 20px; font-style: italic; }}
            .mermaid {{ background: #f9f9f9; padding: 20px; border: 1px dashed #ccc; font-family: monospace; }}
        </style>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
        <div class="container">
            {content}
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

if __name__ == "__main__":
    convert_md_to_html('research_paper.md', 'research_paper.html')
    print("Successfully generated research_paper.html")
