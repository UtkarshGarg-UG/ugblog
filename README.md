# UG's Blog

A technical blog built with Hugo and the PaperMod theme, featuring interactive visualizations and demos.

## Features

- **Hugo Static Site Generator** - Fast, flexible static site
- **PaperMod Theme** - Clean, responsive design
- **Interactive Demos** - Custom HTML/JavaScript visualizations
  - LLM-as-a-Judge demos
  - Bradley-Terry model calculator
  - Entropy explorer
  - And more!

## Local Development

```bash
# Install Hugo (macOS)
brew install hugo

# Run development server
hugo server -D

# Build for production
hugo --gc --minify
```

Visit http://localhost:1313 to preview locally.

## Project Structure

```
.
├── content/          # Blog posts (Markdown)
├── static/           # Static assets (images, HTML demos)
├── themes/PaperMod/  # Hugo theme (git submodule)
├── config.toml       # Hugo configuration
└── netlify.toml      # Deployment configuration
```

## Deployment

This site is configured for easy deployment on Netlify:

1. Push to GitHub (already done!)
2. Connect repository to Netlify
3. Netlify auto-detects Hugo and deploys

## Writing Posts

Create a new post:

```bash
hugo new posts/my-new-post.md
```

Posts are written in Markdown with frontmatter for metadata.

## Interactive Demos

Custom demos are stored in `/static/` and embedded via iframes:

```markdown
<iframe src="/llm-judge/intro.html" width="100%" height="980"></iframe>
```

## License

Content: All rights reserved
Theme: MIT License (PaperMod)
