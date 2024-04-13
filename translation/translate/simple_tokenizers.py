import re

def universal_sentence_tokenizer(text):
    # Define a regular expression pattern for sentence terminators (period, exclamation mark, question mark)
    # We include cases like multiple punctuation (e.g., "?!") and consider spaces after the punctuation.
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
    
    # Use re.split to split the text at each sentence-ending punctuation followed by whitespace
    sentences = re.split(sentence_endings, text.strip())
    
    # Filter out any empty sentences that might be captured as splits
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

def universal_word_tokenizer(text):
    # Define a regular expression pattern to identify word boundaries
    # This pattern splits on spaces, and most common punctuation marks
    word_boundaries = r'[ \s,\.\?!;:\(\)\[\]\"\'\–\—\-\+\=\/\&\%\$\#\@\^\*\|\~\<\>\{\}`]'
    
    # Use re.split to split the text at each occurrence of the pattern
    words = re.split(word_boundaries, text)
    
    # Filter out any empty words that might be captured as splits
    words = [word.strip() for word in words if word.strip()]
    
    return words