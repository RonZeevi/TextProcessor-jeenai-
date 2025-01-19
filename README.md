# Text Processor

`TextProcessor` הוא מודול לעיבוד טקסטים, המאפשר קריאה, חלוקה, יצירת embeddings, ואחסון ושליפה של מסמכים באמצעות `chromadb`.

## תכונות עיקריות

- קריאת קבצים בפורמטים שונים: TXT, PDF, DOCX.
- חלוקת טקסט למקטעים בגודל קבוע, לפי משפטים או פסקאות.
- יצירת embeddings באמצעות מודל BigBird.
- אחסון ושליפה של מסמכים ו-embeddings ממסד נתונים `chromadb`.

## התקנה

1. ודא שיש לך את Python מותקן במערכת שלך.
2. התקן את התלויות הנדרשות באמצעות pip:

   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt**:
   ```
   nltk
   transformers
   chromadb
   pdfplumber
   torch
   numpy
   python-docx
   ```

