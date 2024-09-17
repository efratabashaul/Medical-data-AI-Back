# server.py
from flask import Flask, jsonify, request
import pyodbc
from explore import main
from explore import arrange
from PersonDetailClass import PersonDetails
from dataclasses import asdict
from flask_cors import CORS
from explore import parse_date
from datetime import datetime


app = Flask(__name__)
CORS(app)  # הוסף את התמיכה ב-CORS לכלל האפליקציה


@app.route('/api/process_text', methods=['GET'])
def process_text():
     print(request.args)
    # try:
        # קבלת הטקסט מצד הלקוח
     text = request.args.get('text')  # שימוש ב- request.args לקבלת הפרמטרים מה-URL
     if not text:
        return jsonify({'error': 'No text provided'}), 400
        # שליחת הטקסט לפונקציה ARRANGE
     arranged_object: PersonDetails = arrange(text)
     return jsonify(asdict(arranged_object)), 201


@app.route('/api/process_text', methods=['POST'])
def add():
    try:
        print("reqqq")
        print(request.json)
        data = request.json
        # התחברות למסד הנתונים של SQL Server
        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=(localdb)\\MSSQLLocaldb;'  # שם השרת שלך
            'DATABASE=Injuri;'  # שם מסד הנתונים שלך
            'Trusted_Connection=yes;'
        )
        cursor = connection.cursor()
        print("dateeeeeeeeeeeee")
        print(data['date'])
        date_of_injury = parse_date(data['date'])
        # הוספת האובייקט שחזר מ-ARRANGE לטבלת Injuri
        cursor.execute('''
            INSERT INTO Injuri (name, age, id,nameFather,doctorType,hospital,date) 
            VALUES (?, ?, ?,?,?,?,?)
        ''', data['name'], data['age'], data['id'],data['nameFather'],data['doctorType'],data['hospital'],date_of_injury)
        # שמירת השינויים במסד הנתונים
        connection.commit()
        # סגירת החיבור למסד הנתונים
        connection.close()
        # החזרת האובייקט המעובד לצד הלקוח
        return jsonify(data), 201

    except Exception as e:
        # במקרה של תקלה, נבצע Rollback ונחזיר הודעת שגיאה
        if 'connection' in locals():
            connection.rollback()
            connection.close()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
