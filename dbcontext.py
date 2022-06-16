""" E.g.
    1. Creating a connection to the database and creating a cursor object.
    2. run these only lines
    3. then uncomment the CREATE TABLE PART ONLY and run it then comment the create table part again
    4. then uncomment the INSERT TO DATABASE PART ONLY and run it then comment the insert to database part again
    5. to see  if  database is created and the datas are insert uncomment the SELECT AND PRINT PART , then run it and comment it our again
"""



import sqlite3
conn = sqlite3.connect('AlgoTrade.db')
c = conn.cursor()



# c.execute("""
#           CREATE TABLE User (
#               id integer not null primary key AUTOINCREMENT,
#               name text not null,
#               email text not null,
#               password text not null
#             )
#           """)




# c.execute("INSERT INTO User (name,email,password) VALUES('Souhardya','souhardya@gmail.com','souhardya')")
# c.execute("INSERT INTO User (name,email,password) VALUES('Ritz','ritz@gmail.com','ritz')")
# c.execute("INSERT INTO User (name,email,password) VALUES('Asik','asik@gmail.com','asik')")
# c.execute("INSERT INTO User (name,email,password) VALUES('Usuf','usuf@gmail.com','usuf')")
# c.execute("INSERT INTO User (name,email,password) VALUES('Suprotim','suprotim@gmail.com','suprotim')")
# c.execute("INSERT INTO User (name,email,password) VALUES ('{n}','{e}','{p}')".format(n = "a",e = "abs@gmail.com",p = "abc"))



# c.execute("SELECT * FROM User")
# print(c.fetchall())



conn.commit()
conn.close()
