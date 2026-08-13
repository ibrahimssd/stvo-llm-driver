import re

class LegalTextSegmenter:
    """
    Combines structural markers with semantic understanding.
    """
    def __init__(self, tokenizer=None, max_segment_length=128):
        self.tokenizer = tokenizer
        self.max_segment_length = max_segment_length

        # Legal structure patterns
        self.paragraph_pattern = re.compile(r'\(([0-9]+)\)')
        self.section_pattern = re.compile(r'§\s*([0-9]+)')
        self.subpoint_pattern = re.compile(r'^[0-9]+\.|^[a-z]\)|^[a-z]{2}\)')
        
    def segment_text(self, text):
        """
        Multi-level segmentation:
        1. Split by major sections (§)
        2. Split by paragraphs within sections
        3. Split long paragraphs by sentence boundaries
        """
        # Protect special tokens
        text = self._protect_special_tokens(text)
        
        # Level 1: Split by sections
        sections = self._split_by_sections(text)
        
        # Level 2: Split each section by paragraphs
        all_segments = []
        for section in sections:
            paragraphs = self._split_by_paragraphs(section)
            
            # Level 3: Split long paragraphs
            for para in paragraphs:
                if self._token_count(para) > self.max_segment_length:
                    sub_segments = self._split_long_paragraph(para)
                    all_segments.extend(sub_segments)
                else:
                    all_segments.append(para)
        
        # Restore special tokens
        all_segments = [self._restore_special_tokens(s) for s in all_segments]
        
        return all_segments
    
    def _protect_special_tokens(self, text):
        """Prevent splitting within XML-like tags."""
        text = text.replace('<sign>', '⟨SIGN⟩')
        text = text.replace('</sign>', '⟨/SIGN⟩')
        return text
    
    def _restore_special_tokens(self, text):
        text = text.replace('⟨SIGN⟩', '<sign>')
        text = text.replace('⟨/SIGN⟩', '</sign>')
        return text
    
    def _split_by_sections(self, text):
        """Split by § markers."""
        matches = list(self.section_pattern.finditer(text))
        if not matches:
            return [text]
        
        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            sections.append(text[start:end].strip())
        
        return sections
    
    def _split_by_paragraphs(self, text):
        """Split by (1), (2), (3) markers."""
        matches = list(self.paragraph_pattern.finditer(text))
        if not matches:
            return [text]
        
        paragraphs = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            paragraphs.append(text[start:end].strip())
        
        return paragraphs
    
    def _split_long_paragraph(self, paragraph):
        """Split long paragraphs at sentence boundaries."""
        # Don't split on decimal points or abbreviations
        sentences = re.split(
            r'(?<!\d)\.(?!\d)(?!⟨)(?!\s*m\b)(?!\s*km)\s+(?=[A-Z(])', 
            paragraph
        )
        
        chunks = []
        current = []
        current_tokens = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            tokens = self._token_count(sent)
            
            if current_tokens + tokens > self.max_segment_length and current:
                chunks.append(' '.join(current))
                current = []
                current_tokens = 0
            
            current.append(sent)
            current_tokens += tokens
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def _token_count(self, text):
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

