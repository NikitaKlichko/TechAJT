import re
def clean_text(text):
    # Remove html-tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove non russian
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text