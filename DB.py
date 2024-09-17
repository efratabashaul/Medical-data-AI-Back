import pyodbc

# פרטי החיבור שלך
server = '(localdb)\\MSSQLLocaldb'
database = 'Injuri'
# יצירת חיבור ל-SQL Server
conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};')

# יצירת אובייקט של הקורסור
cursor = conn.cursor()

# def addToDB():
# הוספת נתונים
cursor.execute('''
INSERT INTO Injuri (name, id) VALUES (?, ?)
''', ('John Doe', 326616687))

# שמירה וסיום החיבור
conn.commit()


