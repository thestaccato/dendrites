import os
import sqlite3
from datetime import datetime
import json
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dendrites.db')

class Database:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.connect()
        self._create_tables()

    def _create_tables(self):
        if not self.ensure_connection():
            return
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symptoms TEXT,
                    predicted_disease TEXT,
                    confidence REAL,
                    medications TEXT,
                    precautions TEXT,
                    alternative_diseases TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_id INTEGER REFERENCES recommendation_history(id),
                    is_accurate INTEGER,
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS symptom_disease_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symptom TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    UNIQUE(symptom, disease)
                )
            """)
            self.conn.commit()
        except Exception as e:
            print(f"Error creating tables: {e}")

    def connect(self):
        self.close()
        max_retries = 3
        retry_count = 0
        retry_delay = 1

        while retry_count < max_retries:
            try:
                self.conn = sqlite3.connect(DB_PATH)
                self.conn.row_factory = sqlite3.Row
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA foreign_keys=ON")
                self.cursor = self.conn.cursor()
                print("Database connection established successfully (SQLite)")
                return True
            except Exception as e:
                print(f"Error connecting to the database (attempt {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Failed to connect to the database after multiple attempts")
                    return False

    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            if self.conn:
                self.conn.close()
                self.conn = None
                print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}")

    def ensure_connection(self):
        try:
            if self.conn is None or self.cursor is None:
                return self.connect()
            self.cursor.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database connection lost: {e}")
            return self.connect()

    def save_recommendation(self, symptoms, predicted_disease, confidence,
                            medications, precautions, alternative_diseases):
        if not self.ensure_connection():
            print("Cannot save recommendation: Database connection failed")
            return None
        try:
            alt_diseases_json = json.dumps([
                {"disease": disease, "probability": probability}
                for disease, probability in alternative_diseases
            ])

            self.cursor.execute("""
                INSERT INTO recommendation_history
                (symptoms, predicted_disease, confidence, medications, precautions, alternative_diseases)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (json.dumps(symptoms), predicted_disease, confidence,
                  json.dumps(medications), json.dumps(precautions), alt_diseases_json))
            self.conn.commit()

            rec_id = self.cursor.lastrowid

            for symptom in symptoms:
                self.update_symptom_disease_stats(symptom, predicted_disease)

            return rec_id
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error in save_recommendation: {e}")
            self.ensure_connection()
            return None

    def update_symptom_disease_stats(self, symptom, disease):
        if not self.ensure_connection():
            print("Cannot update symptom-disease stats: Database connection failed")
            return
        try:
            self.cursor.execute(
                "SELECT count FROM symptom_disease_stats WHERE symptom = ? AND disease = ?",
                (symptom, disease)
            )
            result = self.cursor.fetchone()
            if result:
                self.cursor.execute(
                    "UPDATE symptom_disease_stats SET count = count + 1 WHERE symptom = ? AND disease = ?",
                    (symptom, disease)
                )
            else:
                self.cursor.execute(
                    "INSERT INTO symptom_disease_stats (symptom, disease, count) VALUES (?, ?, 1)",
                    (symptom, disease)
                )
            self.conn.commit()
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error in update_symptom_disease_stats: {e}")
            self.ensure_connection()

    def save_feedback(self, recommendation_id, is_accurate, feedback_text=None):
        if not self.ensure_connection():
            print("Cannot save feedback: Database connection failed")
            return None
        try:
            self.cursor.execute("""
                INSERT INTO user_feedback (recommendation_id, is_accurate, feedback_text)
                VALUES (?, ?, ?)
            """, (recommendation_id, is_accurate, feedback_text))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            print(f"Error in save_feedback: {e}")
            self.ensure_connection()
            return None

    def get_history(self, limit=10):
        if not self.ensure_connection():
            print("Cannot get history: Database connection failed")
            return []
        try:
            self.cursor.execute("""
                SELECT * FROM recommendation_history
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            rows = self.cursor.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['symptoms'] = json.loads(d['symptoms']) if isinstance(d['symptoms'], str) else d['symptoms']
                d['medications'] = json.loads(d['medications']) if isinstance(d['medications'], str) else d['medications']
                d['precautions'] = json.loads(d['precautions']) if isinstance(d['precautions'], str) else d['precautions']
                d['alternative_diseases'] = json.loads(d['alternative_diseases']) if isinstance(d['alternative_diseases'], str) else d['alternative_diseases']
                if isinstance(d.get('created_at'), str):
                    try:
                        d['created_at'] = datetime.fromisoformat(d['created_at'])
                    except ValueError:
                        try:
                            d['created_at'] = datetime.strptime(d['created_at'], '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            d['created_at'] = datetime.now()
                result.append(d)
            return result
        except Exception as e:
            print(f"Error in get_history: {e}")
            self.ensure_connection()
            return []

    def get_popular_symptoms(self, limit=10):
        if not self.ensure_connection():
            print("Cannot get popular symptoms: Database connection failed")
            return []
        try:
            self.cursor.execute("""
                SELECT symptom, SUM(count) as total
                FROM symptom_disease_stats
                GROUP BY symptom
                ORDER BY total DESC
                LIMIT ?
            """, (limit,))
            return [(row['symptom'], row['total']) for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"Error in get_popular_symptoms: {e}")
            self.ensure_connection()
            return []

    def get_symptom_statistics(self, symptom):
        if not self.ensure_connection():
            print("Cannot get symptom statistics: Database connection failed")
            return {
                'symptom': symptom,
                'total_occurrences': 0,
                'associated_diseases': []
            }
        try:
            self.cursor.execute("""
                SELECT disease, count
                FROM symptom_disease_stats
                WHERE symptom = ?
                ORDER BY count DESC
            """, (symptom,))
            diseases = [(row['disease'], row['count']) for row in self.cursor.fetchall()]
            total = sum(count for _, count in diseases)
            return {
                'symptom': symptom,
                'total_occurrences': total,
                'associated_diseases': diseases
            }
        except Exception as e:
            print(f"Error in get_symptom_statistics: {e}")
            self.ensure_connection()
            return {
                'symptom': symptom,
                'total_occurrences': 0,
                'associated_diseases': []
            }
