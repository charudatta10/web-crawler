import re

def clean_text(text):
    # Remove extra lines and unwanted spaces
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove single characters
    text = re.sub(r'\b\w\b', '', text)

    text = re.sub(r'\b(?!a\b)\w\b', '', text)

    # Remove misplaced words (optional, based on specific criteria)
    # Example: Remove words with less than 3 characters
    #text = ' '.join([word for word in text.split() if len(word) > 2])
    
    return text

def summarize_text(text):
    # Simple summarization by extracting key sentences
    sentences = text.split('. ')
    summary = '. '.join(sentences[:3])  # Extract first 3 sentences as summary
    return summary

def process_markdown_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    cleaned_content = clean_text(content)
    summary = summarize_text(cleaned_content)
    
    with open(f"cleaned_{filename}", 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
        file.write('\n\n## Summary\n\n')
        file.write(summary)

if __name__ == "__main__":
    filename = "articles.md"
    process_markdown_file(filename)
