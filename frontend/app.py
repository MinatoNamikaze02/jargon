"""
FastAPI backend for privacy policy NER visualization.
Serves a beautiful web interface for analyzing privacy policies with entity recognition.
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.predict import predict as ner_predict


app = FastAPI(title="Privacy Policy NER Analyzer", description="Beautiful visualization of privacy policy entities")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class PolicyRequest(BaseModel):
    text: str
    model_path: str = "../model"  # Default model path


class EntitySpan(BaseModel):
    text: str
    start: int
    end: int
    label: str
    confidence: float


class SentenceResult(BaseModel):
    text: str
    start: int
    end: int
    entities: List[EntitySpan]


class PolicyResponse(BaseModel):
    sentences: List[SentenceResult]
    entity_summary: Dict[str, int]
    total_entities: int


def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences with start/end positions."""
    # Simple sentence splitting - could be enhanced with spaCy/NLTK
    sentences = []
    pattern = r'[.!?]+\s+'
    
    start = 0
    for match in re.finditer(pattern, text):
        end = match.start() + 1
        sentence = text[start:end].strip()
        if sentence:
            sentences.append((sentence, start, end))
        start = match.end()
    
    # Add final sentence if exists
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            sentences.append((sentence, start, len(text)))
    
    return sentences


def process_sentence_entities(sentence: str, model_path: str) -> List[EntitySpan]:
    """Process a single sentence through NER model."""
    try:
        # Use the existing predict function from src/predict.py
        print(sentence)
        predictions = ner_predict(sentence, model_path)
        entities = []
        
        for pred in predictions:
            entities.append(EntitySpan(
                text=pred.get("word", ""),
                start=pred.get("start", 0),
                end=pred.get("end", 0),
                label=pred.get("entity_group", ""),
                confidence=pred.get("score", 0.0)
            ))
        
        return entities
    except Exception as e:
        print(f"Error processing sentence: {e}")
        return []


@app.post("/analyze", response_model=PolicyResponse)
async def analyze_policy(request: PolicyRequest):
    """Analyze privacy policy text and return entity annotations."""
    try:
        # Split into sentences
        sentences = split_into_sentences(request.text)
        
        results = []
        entity_counts = {}
        total_entities = 0
        
        for sentence_text, sent_start, sent_end in sentences:
            # Process sentence through NER
            entities = process_sentence_entities(sentence_text, request.model_path)
            
            # Adjust entity positions to global text coordinates
            adjusted_entities = []
            for entity in entities:
                adjusted_entity = EntitySpan(
                    text=entity.text,
                    start=entity.start + sent_start,
                    end=entity.end + sent_start,
                    label=entity.label,
                    confidence=entity.confidence
                )
                adjusted_entities.append(adjusted_entity)
                
                # Count entities
                label = entity.label
                entity_counts[label] = entity_counts.get(label, 0) + 1
                total_entities += 1
            
            results.append(SentenceResult(
                text=sentence_text,
                start=sent_start,
                end=sent_end,
                entities=adjusted_entities
            ))
        
        return PolicyResponse(
            sentences=results,
            entity_summary=entity_counts,
            total_entities=total_entities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Serve the main HTML interface."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>Privacy Policy Analyzer</title></head>
            <body>
                <h1>Privacy Policy NER Analyzer</h1>
                <p>Interface file not found. Please ensure index.html exists.</p>
            </body>
        </html>
        """)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
