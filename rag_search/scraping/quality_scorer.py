import re
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
from typing import List, Tuple

from rag_search.utils.logging import (
    log_operation_start, log_operation_end, log_info, 
    log_input, log_output, log_success
)


class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


class QualityImprover:
    def __init__(self, verbose: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.config = AutoConfig.from_pretrained("nvidia/quality-classifier-deberta")
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/quality-classifier-deberta")
        self.model = QualityModel.from_pretrained("nvidia/quality-classifier-deberta").to(self.device)
        self.model.eval()
        self.verbose = verbose
        
        # Score mapping
        self.score_dict = {
            'Low': 0,
            'Medium': 1,
            'High': 2
        }
        
        if self.verbose:
            log_info(f"QualityImprover initialized with device: {self.device}", "QualityImprover")

    def predict_quality_scores(self, text_list: List[str]) -> List[float]:
        """
        Predict educational value scores for a list of texts using NVIDIA's DeBERTa model.
        Returns a list of scores between 0 and 2.
        
        Args:
            text_list: List of text strings to evaluate
            
        Returns:
            List of quality scores where:
            0 = Low quality
            1 = Medium quality
            2 = High quality
        """
        if self.verbose:
            log_operation_start("PREDICTING QUALITY SCORES", "QualityImprover")
            log_input(f"Processing {len(text_list)} text segments", "QualityImprover")
        
        # Process inputs through tokenizer
        inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding="longest",
            truncation=True
        ).to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])

        # Convert predictions to scores
        predicted_classes = torch.argmax(outputs, dim=1)
        scores = [float(score) for score in predicted_classes.cpu().numpy()]
        
        if self.verbose:
            log_success(f"Predicted {len(scores)} quality scores", "QualityImprover")
            log_output(f"Scores: {scores}", "QualityImprover")
            log_operation_end("PREDICTING QUALITY SCORES", "QualityImprover")

        return scores
    

    def clean_markdown_links(self, text: str, min_quality_score: float = 0.2) -> Tuple[str, float]:
        """
        Clean markdown links and filter low-quality content.
        Returns tuple of (cleaned_text, quality_score)
        """
        if self.verbose:
            log_operation_start("CLEANING MARKDOWN", "QualityImprover")
            log_input(f"Text length: {len(text)}", "QualityImprover")
        
        # Split by double newlines to preserve paragraph structure
        paragraphs = text.split('\n\n')
        
        if self.verbose:
            log_info(f"Processing {len(paragraphs)} paragraphs", "QualityImprover")
            
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            # Preserve code blocks by checking if paragraph contains ``` tags
            if '```' in paragraph:
                cleaned_paragraphs.append(paragraph)
                continue
                
            lines = paragraph.split('\n')
            filtered_lines = []
            for line in lines:
                line = line.strip()
                # Keep headers regardless of length
                if re.match(r'^#{1,6}\s+', line):
                    filtered_lines.append(line)
                    continue
                
                # Skip common UI/navigation elements
                if re.match(r'^(Share|Trade|More|Buy|Sell|Download|Menu|Home|Back|Next|Previous|\d+\s*(BTC|USD|EUR|GBP)|\w{3}-\w{1,3}|Currency:.*|You (Buy|Spend|Receive)|≈|\d+\.\d+)', line, re.IGNORECASE):
                    continue
                    
                # Count words before removing markdown
                word_count = len(re.sub(r'\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>', '', line).split())
                
                # Increase minimum word threshold to 12
                if word_count < 12:
                    # Check if line only contains markdown patterns or appears to be a currency/trading related line
                    cleaned_line = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)|\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>|\d+(\.\d+)?%?|\$\d+(\.\d+)?', '', line).strip()
                    if not cleaned_line or len(cleaned_line.split()) < 8:  # If nothing substantial remains, skip this line
                        continue
                
                filtered_lines.append(line)
            
            # Only add paragraph if it has any lines left
            if filtered_lines:
                cleaned_paragraphs.append('\n'.join(filtered_lines))
        
        # Rejoin with double newlines
        cleaned_text = '\n\n'.join(cleaned_paragraphs)
        
        # Get quality score
        quality_score = self.predict_educational_value([cleaned_text])[0]
        
        if self.verbose:
            log_success(f"Cleaned text length: {len(cleaned_text)}, Quality score: {quality_score}", "QualityImprover")
            log_operation_end("CLEANING MARKDOWN", "QualityImprover")
        
        return cleaned_text, quality_score

    def filter_quality_content(self, text: str, min_quality_score: float = 0.2) -> str:
        """
        Filter content based on quality and returns concatenated quality content
        """
        if self.verbose:
            log_operation_start("FILTERING QUALITY CONTENT", "QualityImprover")
            log_input(f"Text length: {len(text)}, Min quality score: {min_quality_score}", "QualityImprover")
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        if self.verbose:
            log_info(f"Processing {len(paragraphs)} paragraphs", "QualityImprover")
        
        # Process each paragraph
        quality_content = []
        for paragraph in paragraphs:
            if not paragraph.strip():  # Skip empty paragraphs
                continue
                
            cleaned_text, quality_score = self.clean_markdown_links(paragraph, min_quality_score)
            if cleaned_text and quality_score >= min_quality_score:
                quality_content.append((cleaned_text, quality_score))
        
        # Debug print
        print(f"Found {len(quality_content)} quality paragraphs out of {len(paragraphs)} total")
        
        if self.verbose:
            log_info(f"Found {len(quality_content)} quality paragraphs out of {len(paragraphs)} total", "QualityImprover")
        
        result = text
        if quality_content:
            result = "\n\n".join(text for text, _ in quality_content)
            
        if self.verbose:
            log_success(f"Filtered content length: {len(result)}", "QualityImprover")
            log_operation_end("FILTERING QUALITY CONTENT", "QualityImprover")
            
        return result  # Return original text if no quality content found

    def replace_newlines(text: str) -> str:
        """Replace multiple newlines with a single space."""
        return re.sub("\n+", " ", text)

    def predict_educational_value(self, text_list: List[str]) -> List[float]:
        """
        Predict educational value scores for a list of texts.
        Returns a list of scores between 0 and 2.
        """
        if self.verbose:
            log_operation_start("PREDICTING EDUCATIONAL VALUE", "QualityImprover")
            log_input(f"Processing {len(text_list)} text segments", "QualityImprover")
            
        scores = self.predict_quality_scores(text_list)
        
        if self.verbose:
            log_success(f"Educational value scores: {scores}", "QualityImprover")
            log_operation_end("PREDICTING EDUCATIONAL VALUE", "QualityImprover")
            
        return scores 
