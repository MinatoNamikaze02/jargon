# Privacy Policy NER Analyzer Frontend

A beautiful web interface for analyzing privacy policies using Named Entity Recognition (NER). This frontend provides real-time visualization of entities found in privacy policy text with interactive filtering and stunning visual design.

## Features

🎨 **Beautiful UI/UX**
- Clean, minimalist design with smooth animations
- Dark/Light mode toggle with system preference detection
- Responsive design that works on all devices
- Color-coded entity highlighting with hover tooltips

🔍 **Advanced Analysis**
- Sentence-level NER processing
- Interactive entity filtering
- Real-time statistics and summaries
- Support for custom trained models

⚡ **Performance**
- Fast FastAPI backend
- Concurrent sentence processing
- Efficient caching and error handling
- Optimized for large policy documents

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### 2. Ensure Model is Available

Make sure you have a trained NER model available. The default path is `./model` (relative to the project root).

### 3. Start the Server

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open in Browser

Navigate to `http://localhost:8000` to access the interface.

## Usage

1. **Input Text**: Paste your privacy policy text in the textarea
2. **Model Path**: Optionally specify a custom model path (defaults to `./model`)
3. **Analyze**: Click "Analyze Policy" or use Ctrl+Enter
4. **Explore Results**: 
   - View entity statistics and counts
   - Filter by entity types using the interactive buttons
   - Hover over highlighted entities for confidence scores
   - Toggle dark/light mode for comfortable viewing

## API Endpoints

### `POST /analyze`

Analyze privacy policy text and return entity annotations.

**Request Body:**
```json
{
  "text": "Your privacy policy text here...",
  "model_path": "./model"
}
```

**Response:**
```json
{
  "sentences": [
    {
      "text": "We collect personal information...",
      "start": 0,
      "end": 35,
      "entities": [
        {
          "text": "personal information",
          "start": 11,
          "end": 31,
          "label": "DATA_TYPE",
          "confidence": 0.95
        }
      ]
    }
  ],
  "entity_summary": {
    "DATA_TYPE": 5,
    "ORGANIZATION": 3
  },
  "total_entities": 8
}
```

### `GET /`

Serves the main HTML interface.

## Entity Types & Colors

The interface supports various entity types with distinct color coding:

- 🔴 **PERSON** - Individual names and personal references
- 🟢 **ORG** - Organizations and companies
- 🔵 **GPE** - Geographical and political entities
- 🟠 **DATE** - Dates and temporal expressions
- 🟦 **MONEY** - Monetary amounts and financial terms
- 🟣 **PRODUCT** - Products and services
- 🟡 **LAW** - Legal terms and regulations
- 🔷 **EMAIL** - Email addresses and contact information

## Customization

### Adding New Entity Types

1. Update the CSS color definitions in `index.html`:
```css
.entity-CUSTOM, .entity-filter[data-entity="CUSTOM"].active { 
  background: #your-color; 
  border-color: #your-color; 
}
```

2. The backend will automatically handle new entity types returned by your model.

### Styling

The interface uses CSS custom properties for easy theming. Key variables:

```css
:root {
  --primary-bg: #ffffff;
  --accent-primary: #667eea;
  --text-primary: #1a202c;
  /* ... */
}
```

## Development

### Project Structure

```
frontend/
├── app.py              # FastAPI backend
├── index.html          # Main interface (self-contained)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Features Implementation

- **Sentence Splitting**: Uses regex-based sentence boundary detection
- **Entity Alignment**: Adjusts entity positions from sentence-relative to document-relative coordinates
- **Concurrent Processing**: Processes multiple sentences in parallel for speed
- **Interactive Filtering**: Real-time filtering without re-processing
- **Responsive Design**: Works on mobile, tablet, and desktop

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure your trained model exists at the specified path
2. **Import Errors**: Make sure the parent `src` directory is in Python path
3. **Port Already in Use**: Change the port in `app.py` or kill existing processes

### Performance Tips

- For large documents, consider chunking text before sending to the API
- Use a smaller model for faster inference if accuracy isn't critical
- Enable caching in your NER pipeline for repeated analyses

## License

This frontend is part of the TNC (Privacy Policy NER) project.
