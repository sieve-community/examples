import re

def universal_sentence_tokenizer(text):
    # Define a regular expression pattern for sentence terminators (period, exclamation mark, question mark)
    # We include cases like multiple punctuation (e.g., "?!") and consider spaces after the punctuation.
    # Enhanced to better handle non-Latin scripts like Chinese, Japanese, and Arabic where spaces might not be used.
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|。|！|？|\u3002)\s*'
    
    # Use re.split to split the text at each sentence-ending punctuation followed by optional whitespace
    sentences = re.split(sentence_endings, text.strip())
    
    # Filter out any empty sentences that might be captured as splits
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

def universal_word_tokenizer(text):
    # Define a regular expression pattern to identify word boundaries
    # This pattern splits on spaces, most common punctuation marks, and enhances support for non-Latin scripts
    # by including common CJK (Chinese, Japanese, Korean) characters and Arabic script boundaries.
    word_boundaries = r'[ \s,\.\?!;:\(\)\[\]\"\'\–\—\-\+\=\/\&\%\$\#\@\^\*\|\~\<\>\{\}`' \
                      r'|\u3000|\u3001|\u3002|\uff0c|\uff01|\uff1f|\u060c|\u061b|\u061f]'
    
    # Use re.split to split the text at each occurrence of the pattern
    words = re.split(word_boundaries, text)
    
    # Filter out any empty words that might be captured as splits
    words = [word.strip() for word in words if word.strip()]
    
    return words