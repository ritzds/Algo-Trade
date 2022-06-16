c.execute("""
          CREATE TABLE User (
              id integer not null primary key AUTOINCREMENT,
              name text not null,

              email text not null,
              password text not null
            )
          """)