import nltk
from typing import List, Tuple
from docx import Document
import os
from transformers import BigBirdModel, BigBirdTokenizer
import chromadb
from datetime import datetime
import uuid
import pdfplumber
import torch
import numpy as np

class TextProcessor:
    def __init__(self):
        """אתחול המעבד"""
        # טעינת משאבי NLTK
        print("מוריד משאבי NLTK...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # אתחול המודל ליצירת embeddings
        print("טוען מודל Embedding (BigBird)...")
        self.tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        self.model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
        
        # הגדרת המודל למצב הערכה לחיסכון בזיכרון
        self.model.eval()
        
        # אתחול מסד הנתונים
        db_path = "C:/Users/ronze/PycharmProjects/jeenaiAssignment/chromadb"
        self.db = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.db.create_collection(name="test")
        except Exception as e:
            self.collection = self.db.get_collection(name="test")

    def read_file(self, file_path: str) -> str:
        """קריאת קובץ טקסט בפורמטים שונים"""
        if not os.path.exists(file_path):
            raise Exception(f"הקובץ לא נמצא: {file_path}")
            
        file_type = file_path.split('.')[-1].lower()
        
        if file_type == 'txt':
            return self._read_txt(file_path)
        elif file_type == 'pdf':
            return self._read_pdf(file_path)
        elif file_type == 'docx':
            return self._read_docx(file_path)
        else:
            raise Exception(f"פורמט קובץ לא נתמך: {file_type}")
    
    def _read_txt(self, file_path: str) -> str:
        """TXT קריאת קובץ"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"שגיאה בקריאת TXT: {str(e)}")
    
    def _read_pdf(self, file_path: str) -> str:
        """PDF קריאת קובץ"""
        try:
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    content = page.extract_text()
                    if content:
                        if len(pdf.pages) > 1:
                            text_parts.append(f"\n[עמוד {page_num}]\n{content}")
                        else:
                            text_parts.append(content)
                        
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"שגיאה בקריאת PDF: {str(e)}")
    
    def _read_docx(self, file_path: str) -> str:
        """DOCX קריאת קובץ"""
        try:
            doc = Document(file_path)
            paragraphs = []
            
            def fix_mixed_word(word):
                """טיפול במילה שמכילה גם עברית וגם אנגלית"""
                current_group = []
                groups = []
                is_prev_hebrew = None
                
                for char in word:
                    is_hebrew = '\u0590' <= char <= '\u05FF'
                    
                    if is_prev_hebrew is None:
                        current_group.append(char)
                    elif is_hebrew != is_prev_hebrew:
                        groups.append(''.join(current_group))
                        current_group = [char]
                    else:
                        current_group.append(char)
                        
                    is_prev_hebrew = is_hebrew
                    
                if current_group:
                    groups.append(''.join(current_group))
                
                fixed_groups = []
                for group in groups:
                    if any('\u0590' <= c <= '\u05FF' for c in group):
                        fixed_groups.append(group[::-1])
                    else:
                        fixed_groups.append(group)
                        
                return ''.join(fixed_groups)

            def is_hebrew_letter_with_dot(word):
                """בדיקה האם זו אות עברית עם נקודה"""
                return (len(word) == 2 and 
                        '\u0590' <= word[0] <= '\u05FF' and 
                        word[1] == '.')

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    words = paragraph.text.split()
                    fixed_words = []
                    
                    i = 0
                    while i < len(words):
                        # בדיקה לסימון חלק
                        if i < len(words) - 1 and words[i] == 'חלק' and len(words[i+1]) == 1:
                            fixed_words.append(f"חלק {words[i+1]}")
                            i += 2
                            continue
                        
                        # בדיקה לאות מספור
                        if is_hebrew_letter_with_dot(words[i]):
                            fixed_words.append(words[i])
                            i += 1
                            continue
                        
                        # טיפול רגיל במילה
                        word = words[i]
                        if any('\u0590' <= c <= '\u05FF' for c in word):
                            if any(c.isascii() for c in word):
                                fixed_words.append(fix_mixed_word(word))
                            else:
                                fixed_words.append(word[::-1])
                        else:
                            fixed_words.append(word)
                        i += 1
                    
                    paragraphs.append(' '.join(fixed_words))
            
            return '\n\n'.join(paragraphs)
            
        except Exception as e:
            raise Exception(f"שגיאה בקריאת DOCX: {str(e)}")
    
    def split_by_sentences(self, text: str) -> List[str]:
        """חלוקה למשפטים"""
        if not text:
            return []
            
        # חלוקה למשפטים באמצעות NLTK
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """חלוקה לפי פסקאות"""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    def store_embeddings(self, text_id: str, chunks: List[str], embeddings: List[List[float]]):
        """אחסון המקטעים וה-embeddings בדאטה בייס"""
        # יצירת אוסף חדש או שימוש באוסף קיים
        collection = self.db.create_collection(name=f"text_{text_id}")
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # הוספת המקטע וה-embedding לאוסף
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{text_id}_chunk_{i}"]
            )
    
    def split_by_fixed_size(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """חלוקת טקסט למקטעים בגודל קבוע עם חפיפה"""
        if not text or chunk_size <= 0 or overlap >= chunk_size:
            return []
        
        step_size = chunk_size - overlap
        chunks = []
        for i in range(0, len(text), step_size):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
                print(f"נוצר מקטע {len(chunks)}: '{chunk}'")
        
        return chunks
    
    def process_text(self, text: str, split_method: str = 'fixed', **kwargs) -> Tuple[List[str], List[List[float]]]:
        """עיבוד טקסט מלא - מחלוקה ועד embeddings"""
        print("מתחיל חלוקת טקסט...")
        
        if split_method == 'fixed':
            chunks = self.split_by_fixed_size(text, **kwargs)
        elif split_method == 'sentences':
            chunks = self.split_by_sentences(text)
        elif split_method == 'paragraphs':
            chunks = self.split_by_paragraphs(text)
        else:
            raise ValueError(f"שיטת חלוקה לא תקינה: {split_method}")
        
        print(f"נוצרו {len(chunks)} מקטעים")
        
        if not chunks:
            return [], []
        
        print("מתחיל יצירת embeddings...")
        embeddings = self.create_embeddings(chunks)
        print("embeddings נוצרו בהצלחה")
        
        return chunks, embeddings
    
    def create_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """יצירת embeddings למקטעי טקסט"""
        if not chunks:
            return []
        
        embeddings = []
        for chunk in chunks:
            # טוקניזציה והעברה למודל
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            
            # יצירת embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
                embeddings.append(embedding)
                
            # שחרור זיכרון
            del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return embeddings
    
    def _clean_text(self, text: str) -> str:
        """ניקוי וסידור הטקסט"""
        if not text:
            return ""
        
        # הסרת רווחים מיותרים
        lines = text.splitlines()
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
        
    def _clean_pdf_text(self, text: str) -> str:
        """ניקוי טקסט מ-PDF"""
        return self._clean_text(text)
    
    def _fix_hebrew_text(self, text: str) -> str:
        """תיקון כיוון טקסט בעברית"""
        if not text:
            return ""

        # פיצול לשורות
        lines = text.split('\n')
        fixed_lines = []

        for line in lines:
            # טיפול מיוחד במטא-דאטה
            if "{'added_at'" in line:
                fixed_lines.append(line)
                continue

            # טיפול מיוחד במספרי עמודים
            if '[עמוד' in line:
                page_num = line.split('[עמוד')[1].split(']')[0].strip()
                fixed_lines.append(f"[עמוד {page_num}]")
                continue

            # בדיקה האם השורה מכילה טקסט בעברית
            has_hebrew = any('\u0590' <= c <= '\u05FF' for c in line)

            if has_hebrew:
                # מפצלים את השורה לפי מילים
                words = line.split()
                # הופכים רק את המילים בעברית
                fixed_words = []
                for word in words:
                    if any('\u0590' <= c <= '\u05FF' for c in word):
                        fixed_words.append(word[::-1])
                    else:
                        fixed_words.append(word)
                fixed_lines.append(' '.join(fixed_words))
            else:
                fixed_lines.append(line)  # לא הופכים טקסט באנגלית

        return '\n'.join(fixed_lines)

    def fix_rtl_text(self, text):
        """תיקון כיוון טקסט עברי"""
        if not text:
            return text

        def is_hebrew(text):
            # בדיקה אם יש עברית בטקסט
            return any('\u0590' <= c <= '\u05FF' for c in text)

        def fix_line(line):
            # טיפול בשורה בודדת
            if not line.strip():
                return line
            
            # שמירה על פורמט של מספרי עמודים
            if line.strip().startswith('[') and line.strip().endswith(']'):
                return line

            # פיצול לפי רווחים
            words = line.split()
            fixed_words = []
            english_sequence = []

            for word in words:
                # אם זו מילה באנגלית
                if not is_hebrew(word) and word.isalnum():
                    english_sequence.append(word)
                else:
                    # אם יש מילים באנגלית שחיכו בתור
                    if english_sequence:
                        fixed_words.extend(english_sequence)
                        english_sequence = []
                    fixed_words.append(word)

            # הוספת מילים באנגלית שנשארו
            if english_sequence:
                fixed_words.extend(english_sequence)

            # הפיכת סדר המילים אם יש עברית בשורה
            if any(is_hebrew(word) for word in fixed_words):
                fixed_words.reverse()

            return ' '.join(fixed_words)

        # טיפול בכל שורה בנפרד
        lines = text.split('\n')
        fixed_lines = [fix_line(line) for line in lines]
        
        return '\n'.join(fixed_lines)

    def show_database_info(self):
        """הצגת מידע על המסמכים בדאטהבייס"""
        try:
            results = self.collection.get()

            print("\n=== מידע על הדאטהבייס ===")
            if not results['ids']:
                print("אין מסמכים בדאטהבייס")
                return None

            print(f"מספר מסמכים: {len(results['ids'])}")

            for i, doc_id in enumerate(results['ids']):
                print(f"\nמסמך {i+1}:")
                print(f"מזהה: {doc_id}")
                text_preview = results['documents'][i][:100]
                print(f"טקסט: {text_preview}...")
                if results['metadatas'][i]:
                    print("מטא-דאטה:")
                    for key, value in results['metadatas'][i].items():
                        if key == 'added_at':
                            try:
                                dt = datetime.fromisoformat(value)
                                print(f"  נוסף בתאריך: {dt.strftime('%d/%m/%Y %H:%M:%S')}")
                            except:
                                print(f"  {key}: {value}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print("מטא-דאטה: אין")

            return results

        except Exception as e:
            print(f"שגיאה בקבלת מידע מהדאטהבייס: {str(e)}")
            return None
    
    def add_to_database(self, text, metadata=None):
        """הוספת טקסט לדאטהבייס"""
        try:
            doc_id = str(uuid.uuid4())
            
            if metadata is None:
                metadata = {}
            
            metadata['added_at'] = datetime.now().isoformat()
            
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return doc_id
            
        except Exception as e:
            print(f"שגיאה בהוספה לדאטהבייס: {str(e)}")
            return None

    def get_document_by_id(self, doc_id: str) -> str:
        """שליפת מסמך לפי מזהה"""
        try:
            # שליפת המסמך מהאוסף לפי מזהה
            result = self.collection.get(ids=[doc_id])
            
            if not result['documents']:
                print(f"לא נמצא מסמך עם מזהה: {doc_id}")
                return ""
            
            # החזרת תוכן המסמך
            return result['documents'][0]
        
        except Exception as e:
            print(f"שגיאה בשליפת המסמך: {str(e)}")
            return ""