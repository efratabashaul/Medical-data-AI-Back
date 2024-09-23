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
CORS(app)

@app.route('/api/process_text', methods=['GET'])
def process_text():
     print(request.args)
     text = request.args.get('text')  # שימוש ב- request.args לקבלת הפרמטרים מה-URL
     if not text:
        return jsonify({'error': 'No text provided'}), 400
     arranged_object: PersonDetails = arrange(text)
     return jsonify(asdict(arranged_object)), 201


@app.route('/api/process_text', methods=['POST'])
def add():
    try:
        print("reqqq")
        print(request.json)
        data = request.json

        age = data['age']
        if isinstance(age, str) and age.isdigit():
            age = int(age)
        else:
            age = None

        # בדיקה אם id מכיל רק מספרים, אחרת הוא יהיה None
        id_number = data['id']
        if isinstance(id_number, str) and id_number.isdigit():
            id_number = int(id_number)
        else:
            id_number = None

        connection = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=(localdb)\\MSSQLLocaldb;' 
            'DATABASE=Injuri;'  
            'Trusted_Connection=yes;'
        )
        cursor = connection.cursor()
        print("dateeeeeeeeeeeee")
        print(data['date'])
        date_of_injury = parse_date(data['date'])

        cursor.execute('''
            INSERT INTO Injuri (name, age, id,nameFather,doctorType,hospital,date) 
            VALUES (?, ?, ?,?,?,?,?)
        ''', data['name'], age, id_number,data['nameFather'],data['doctorType'],data['hospital'],date_of_injury)
        connection.commit()
        connection.close()
        return jsonify(data), 201

    except Exception as e:
        if 'connection' in locals():
            connection.rollback()
            connection.close()
        return jsonify({'error': str(e)}), 500

#if __name__ == '__main__':
   # app.run(debug=True)
if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)
